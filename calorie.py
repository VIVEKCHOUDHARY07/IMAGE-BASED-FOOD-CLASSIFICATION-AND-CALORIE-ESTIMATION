import cv2
import numpy as np
from image_segment import getAreaOfFood  # Make sure you have the migrated segmentation script

density_dict = {1:0.609, 2:0.94, 3:0.641, 4:0.641, 5:0.513, 6:0.482, 7:0.481}
calorie_dict = {1:52, 2:89, 3:41, 4:16, 5:40, 6:47, 7:18}
skin_multiplier = 5 * 2.3

def getCalorie(label, volume):
    calorie = calorie_dict[int(label)]
    density = density_dict[int(label)]
    mass = volume * density
    calorie_tot = (calorie / 100.0) * mass
    return mass, calorie_tot, calorie

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
    area_fruit = (area / skin_area) * skin_multiplier
    label = int(label)
    volume = 100
    if label in [1, 5, 6, 7]:
        radius = np.sqrt(area_fruit / np.pi)
        volume = (4/3) * np.pi * radius ** 3
    elif label in [2, 4] or (label == 3 and area_fruit > 30):
        fruit_rect = cv2.minAreaRect(fruit_contour)
        height = max(fruit_rect[1]) * pix_to_cm_multiplier
        radius = area_fruit / (2.0 * height)
        volume = np.pi * radius ** 2 * height
    elif (label == 4 and area_fruit < 30):
        volume = area_fruit * 0.5
    return max(volume, 1.0)

def calories(result, img):
    fruit_areas, final_f, areaod, skin_areas, fruit_contours, pix_cm = getAreaOfFood(img)
    volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    return cal
