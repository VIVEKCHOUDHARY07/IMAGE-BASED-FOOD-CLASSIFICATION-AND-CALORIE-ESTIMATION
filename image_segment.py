import cv2
import numpy as np
import os


def getAreaOfFood(img1):
    data = os.path.join(os.getcwd(), "images")
    if os.path.exists(data):
        print('folder exist for images at ', data)
    else:
        os.mkdir(data)
        print('folder created for images at ', data)

    cv2.imwrite(f'{data}/1 original image.jpg', img1)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{data}/2 original image BGR2GRAY.jpg', img)
    img_filt = cv2.medianBlur(img, 5)
    cv2.imwrite(f'{data}/3 img_filt.jpg', img_filt)
    img_th = cv2.adaptiveThreshold(img_filt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    cv2.imwrite(f'{data}/4 img_th.jpg', img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    if len(largest_areas) == 0:
        raise RuntimeError("No contours found in image.")
    cv2.drawContours(mask, [largest_areas[-1]], 0, (255, 255, 255, 255), -1)
    cv2.imwrite(f'{data}/5 mask.jpg', mask)
    img_bigcontour = cv2.bitwise_and(img1, img1, mask=mask)
    cv2.imwrite(f'{data}/6 img_bigcontour.jpg', img_bigcontour)

    hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f'{data}/7 hsv_img.jpg', hsv_img)
    mask_plate = cv2.inRange(hsv_img, np.array([0, 0, 50]), np.array([200, 90, 250]))
    cv2.imwrite(f'{data}/8 mask_plate.jpg', mask_plate)
    mask_not_plate = cv2.bitwise_not(mask_plate)
    cv2.imwrite(f'{data}/9 mask_not_plate.jpg', mask_not_plate)
    fruit_skin = cv2.bitwise_and(img_bigcontour, img_bigcontour, mask=mask_not_plate)
    cv2.imwrite(f'{data}/10 fruit_skin.jpg', fruit_skin)

    hsv_img = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
    cv2.imwrite(f'{data}/11 hsv_img.jpg', hsv_img)
    skin = cv2.inRange(hsv_img, np.array([0, 10, 60]), np.array([10, 160, 255]))
    cv2.imwrite(f'{data}/12 skin.jpg', skin)
    not_skin = cv2.bitwise_not(skin)
    cv2.imwrite(f'{data}/13 not_skin.jpg', not_skin)
    fruit = cv2.bitwise_and(fruit_skin, fruit_skin, mask=not_skin)
    cv2.imwrite(f'{data}/14 fruit.jpg', fruit)

    fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{data}/15 fruit_bw.jpg', fruit_bw)
    fruit_bin = cv2.inRange(fruit_bw, 10, 255)
    cv2.imwrite(f'{data}/16 fruit_bin.jpg', fruit_bin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erode_fruit = cv2.erode(fruit_bin, kernel, iterations=1)
    cv2.imwrite(f'{data}/17 erode_fruit.jpg', erode_fruit)

    img_th = cv2.adaptiveThreshold(erode_fruit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f'{data}/18 img_th.jpg', img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    if len(largest_areas) < 2:
        fruit_idx = -1
    else:
        fruit_idx = -2
    cv2.drawContours(mask_fruit, [largest_areas[fruit_idx]], 0, (255, 255, 255), -1)
    cv2.imwrite(f'{data}/19 mask_fruit.jpg', mask_fruit)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_fruit2 = cv2.dilate(mask_fruit, kernel2, iterations=1)
    cv2.imwrite(f'{data}/20 mask_fruit2.jpg', mask_fruit2)
    fruit_final = cv2.bitwise_and(img1, img1, mask=mask_fruit2)
    cv2.imwrite(f'{data}/21 fruit_final.jpg', fruit_final)

    img_th = cv2.adaptiveThreshold(mask_fruit2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f'{data}/22 img_th.jpg', img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_areas = sorted(contours, key=cv2.contourArea)
    if len(largest_areas) < 2:
        fruit_contour_idx = -1
    else:
        fruit_contour_idx = -2
    fruit_contour = largest_areas[fruit_contour_idx]
    fruit_area = cv2.contourArea(fruit_contour)

    skin2 = skin - mask_fruit2
    cv2.imwrite(f'{data}/23 skin2.jpg', skin2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_e = cv2.erode(skin2, kernel, iterations=1)
    cv2.imwrite(f'{data}/24 skin_e.jpg', skin_e)
    img_th = cv2.adaptiveThreshold(skin_e, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f'{data}/25 img_th.jpg', img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_skin = np.zeros(skin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    if len(largest_areas) < 2:
        skin_idx = -1
    else:
        skin_idx = -2
    cv2.drawContours(mask_skin, [largest_areas[skin_idx]], 0, (255, 255, 255), -1)
    cv2.imwrite(f'{data}/26 mask_skin.jpg', mask_skin)

    skin_rect = cv2.minAreaRect(largest_areas[skin_idx])
    box = cv2.boxPoints(skin_rect)
    box = np.int64(box)  # fixed from deprecated np.int0
    mask_skin2 = np.zeros(skin.shape, np.uint8)
    cv2.drawContours(mask_skin2, [box], 0, (255, 255, 255), -1)
    cv2.imwrite(f'{data}/27 mask_skin2.jpg', mask_skin2)

    pix_height = max(skin_rect[1])
    pix_to_cm_multiplier = 5.0 / pix_height if pix_height != 0 else 0
    skin_area = cv2.contourArea(box)

    return fruit_area, fruit_bin, fruit_final, skin_area, fruit_contour, pix_to_cm_multiplier


if __name__ == '__main__':
    img1 = cv2.imread(r"C:\Users\piya\Desktop\model2\Orange\2.jpg")
    img = cv2.resize(img1, (1000, 1000))
    area, bin_fruit, img_fruit, skin_area, fruit_contour, pix_to_cm_multiplier = getAreaOfFood(img)
    cv2.imshow('img', img_fruit)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
