from math import floor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def cal_utilization(x, y, xrt, yrt, kerfx = 0.07, kerfy = 0.07):
    Us = []
    for i in range(2):
        if i == 0:
            xx, yy = x, y
        else:
            xx, yy = y, x
        xrt -= kerfx
        yrt -= kerfy
        
        ax = floor(xrt/xx)
        ay = floor(yrt/yy)
        
        Areticle = xrt*yrt
        Us.append((ax*xx*ay*yy)/Areticle)
    
    U = Us[0]
    if Us[1] > Us[0]:
        U = Us[1]

    return U*100

def cal_chips(x, y, xrt, yrt, kerfx = 0.07, kerfy = 0.07):
    chips = []
    for i in range(2):
        if i == 0:
            xx, yy = x, y
        else:
            xx, yy = y, x
        xrt -= kerfx
        yrt -= kerfy
        
        ax = floor(xrt/xx)
        ay = floor(yrt/yy)
        
        Areticle = xrt*yrt
        chips.append(ax*ay)
    
    chip = chips[0]
    if chips[1] > chips[0]:
        chip = chips[1]

    return chip

# Utilization function if the chip is allowed to rotate
def cal_utilization_rot(x, y, ax, ay, xrt, yrt, kerfx = 0.07, kerfy = 0.07):
    xrt -= kerfx
    yrt -= kerfy
    
    Areticle = xrt*yrt
    U = (ax*x*ay*y)/Areticle
    
    return U*100



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2