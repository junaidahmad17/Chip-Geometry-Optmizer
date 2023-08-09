
import os
import winsound
import numpy as np
import tensorflow as tf
from math import floor
from datetime import datetime
from scipy.optimize import differential_evolution
from flask import Flask, render_template, request, jsonify
from utils import cal_chips
from dotenv import load_dotenv

load_dotenv()

X1 = os.getenv('X1')
X2 = os.getenv('X2')
Y1 = os.getenv('Y1')
Y2 = os.getenv('Y2')
nc_model_path = os.getenv('NO_OF_CHIPS_MODEL_PATH')
ron_model_path = os.getenv('RON_MODEL_PATH')

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate')
def numChips():
    return render_template('cal_chips.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    # Get the form data
    chip_raster_x = request.json['chipRasterX']
    chip_raster_y = request.json['chipRasterY']
    reticleX, reticleY = int(request.json['reticleSize'][:2]), int(request.json['reticleSize'][3:])

    ron_tol = request.json['ronTol']
    max_iter = request.json['maxIter']
    startTime = request.json['startTime']
    print('start!!!')

    # Load Ron model
    model_ron = tf.keras.models.load_model(ron_model_path)
    # set the bounds for which the Ron model works
    bounds = [(X1, Y1), (X2, Y2)]

    curr_ron = model_ron.predict(np.asarray([[chip_raster_x, chip_raster_y]]), verbose=None)[0][0]
    curr_utilization = cal_chips(chip_raster_x, chip_raster_y, reticleX, reticleY)
    print('curr_ron: ', curr_ron)
    print('curr_utilization: ', curr_utilization)
    # Objective Function
    def objective_function(params):
        x, y = params 

        input_data_ron = np.asarray([[x, y]])

        ret_utilization = cal_chips(x, y, reticleX, reticleY)
        prediction_ron = model_ron.predict(input_data_ron, verbose=None)[0][0]
        fitness_score = -ret_utilization  

        if x < y:
            return 100   
        

        if 100*abs((prediction_ron-curr_ron)/((prediction_ron+curr_ron)/2)) <= ron_tol:
            return fitness_score
        elif 100*abs((prediction_ron-curr_ron)/((prediction_ron+curr_ron)/2)) <= ron_tol+1:
            return fitness_score+1000 
        elif 100*abs((prediction_ron-curr_ron)/((prediction_ron+curr_ron)/2)) <= ron_tol+2:
            return fitness_score+2000 
        elif 100*abs((prediction_ron-curr_ron)/((prediction_ron+curr_ron)/2)) <= ron_tol+3:
            return fitness_score+3000 
        elif 100*abs((prediction_ron-curr_ron)/((prediction_ron+curr_ron)/2)) <= ron_tol+4:
            return fitness_score+4000
        else:
            return 100  

    diff_opt_ron, diff_opt_count = 0, 0
    best_ron_, best_opt_ron, best_count_ron = -100, [-100, -100], -1
    best_ron_count, best_opt_count, best_count_ = -100, [-100, -100], -1
    # bounds for x and y
    print('start iterations!!')
    for i in range(max_iter):
        result = differential_evolution(objective_function, bounds, maxiter=50) # , popsize=30, mutation=1.5, recombination=0.7)
        print('DE Done!!!')
        # Best Solution
        best_solution = result.x
        best_fitness = result.fun

        Ron = float(model_ron.predict(np.asarray([[best_solution[0], best_solution[1]]]), verbose=None)[0][0])
        utilization = cal_chips(best_solution[0], best_solution[1], reticleX, reticleY)

        prec_diff_ron = 100*abs((Ron-curr_ron)/((Ron+curr_ron)/2))

        if abs(curr_ron-Ron) < abs(curr_ron-best_ron_):
            best_ron_ = Ron
            best_opt_ron = best_solution
            best_count_ron = utilization
            diff_opt_ron = round(prec_diff_ron,2)
            

        if (curr_ron*(1-(ron_tol/100)) <= Ron <= curr_ron*(1+(ron_tol/100))) and (best_count_ < utilization):
            best_ron_count = Ron
            best_opt_count = best_solution
            best_count_ = utilization
            diff_opt_count = round(prec_diff_ron, 2)

        print('iteration: ', i+1, ' Done!')
    # Optimal based on Ron 
    x_opt_ron = float(best_opt_ron[0])  
    y_opt_ron = float(best_opt_ron[1])  
    ron_opt_ = best_ron_
    num_chips_ron = best_count_ron 

    # Optimal based on chip count
    print(best_opt_count)
    x_opt_count = float(best_opt_count[0])  
    y_opt_count = float(best_opt_count[1])  
    ron_opt_count = best_ron_count
    num_chips_ = best_count_ 

    # beep before sending response
    winsound.Beep(2500, 5000)
    # Response data prep
    response_data = {
        'xOpt_best_ron': str(x_opt_ron), 
        'yOpt_best_ron': str(y_opt_ron),  
        'ronOpt_best_ron': str(ron_opt_),  
        'numChips_best_ron': str(num_chips_ron),  
        'diff_opt_ron': str(diff_opt_ron),

        'xOpt_best_count': str(x_opt_count), 
        'yOpt_best_count': str(y_opt_count),  
        'ronOpt_best_count': str(ron_opt_count),  
        'numChips_best_count': str(num_chips_), 
        'diff_opt_count': str(diff_opt_count),

        'curr_ron': str(curr_ron),
        'curr_utilization': str(curr_utilization),
        'startTime': str(startTime),  
    }
    print(response_data)
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)
