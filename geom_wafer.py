from math import sqrt
import numpy as np

kerfx = 0.07
kerfy = 0.07
notch = 5
def half_circle_chips(p, r, chip_width, chip_height, bottom=False):
    f = False
    total_chips = 0
    while True:
        if f:
            p = p + (chip_height) + kerfy
            if ~bottom and (p + (chip_height) + kerfy) >= r:
                break
            if bottom and (p + (chip_height) + kerfy) >= (r-notch):
                break
        
        upper_bound = sqrt(r**2 - (p)**2)*2
        
        chip_count = upper_bound // (chip_width + kerfx)

        total_chips += chip_count
        f = True

    return total_chips


def max_chips_in_wafer(r, chip_width, chip_height):
    
    most_chips, best_shift = [0], [0]
    shifts = np.arange(0, 1, 0.01)
    for _ in range(2):
        for shift in shifts:
            p_top = (chip_height/2)-shift
            p_bottom = (chip_height/2)+shift

            top_half = half_circle_chips(p_top, r, chip_width, chip_height)
            bottom_half = half_circle_chips(p_bottom, r, chip_width, chip_height, bottom=True)
            total_chips = top_half + bottom_half

            if most_chips[0] <= total_chips:
                if most_chips[0] < total_chips:
                    most_chips, best_shift = [], []
                
                most_chips.append(int(total_chips))
                best_shift.append(shift)

        shifts = [-x for x in shifts]

    return most_chips

while True:
    print('Enter values or enter -1 as chip height to exit.')
    wafer_radius = float(input('Enter wafer radius: '))
    chip_width = float(input('Enter chip width: '))
    chip_height = float(input('Enter chip height: '))
    
    if chip_height < 0:
        break 

    chips = max_chips_in_wafer(wafer_radius, chip_width, chip_height)
    rot_chips = max_chips_in_wafer(wafer_radius, chip_height, chip_width)

    rot = False
    if rot_chips > chips:
        chips = rot_chips
        rot = True

    print("Maximum chips that can fit in the wafer: ", chips[0])
    print('Rotation: ', rot)

