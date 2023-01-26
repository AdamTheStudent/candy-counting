import json
from pathlib import Path
from typing import Dict
import numpy as np
import click
import cv2
from tqdm import tqdm



def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #################################################
    #Czerwony+ kernel+piersze testy
    #################################################

    # upper mask (170-180)
    lower_red = np.array([170, 50, 160])
    upper_red = np.array([180, 255, 255])
    maskred = cv2.inRange(img_hsv, lower_red, upper_red)
    #mask = mask1

    #cv2.imwrite("red_image1.jpg", mask1)
    #result = img.copy()
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image = cv2.GaussianBlur(image, (11, 11), 0)
    #lower = np.array([175, 200, 0])
    #upper = np.array([200, 230, 255])
    #mask = cv2.inRange(image, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    maskred = cv2.morphologyEx(maskred, cv2.MORPH_OPEN, kernel)
    maskred = cv2.dilate(maskred, kernel, iterations=4)
    (cnt, hierarchy) = cv2.findContours(
        maskred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    red = len(cnt)

    ####################################################
    #Zielony
    #####################################################
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv=cv2.GaussianBlur(img_hsv, (7, 7), 1.5)

    lower_green = np.array( [35,192,84])
    upper_green = np.array( [62,255,191])

    maskgreen = cv2.inRange(img_hsv, lower_green, upper_green)
    cv2.imwrite("green_image1.jpg", maskgreen)
    maskgreen = cv2.erode(maskgreen, kernel, iterations=1)
    cv2.imwrite("green_image2.jpg", maskgreen)
    maskgreen = cv2.morphologyEx(maskgreen, cv2.MORPH_OPEN, kernel)
    #cv2.imwrite("green_image2.jpg", maskgreen)
    maskgreen = cv2.dilate(maskgreen, kernel, iterations=2)

    #img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower_yellowgreen = np.array([20, 120, 50])
    #upper_yellowgreen= np.array([50, 250, 130])

    #maskzoltygreen = cv2.inRange(img_hsv, lower_yellowgreen, upper_yellowgreen)
    #cv2.imwrite("green_image1.jpg", maskzoltygreen)
    #maskzoltygreen = cv2.erode(maskzoltygreen, kernel, iterations=4)
    #cv2.imwrite("green_image2.jpg", maskzoltygreen)
    #maskzoltygreen = cv2.morphologyEx(maskzoltygreen, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("green_image2.jpg", maskgreen)
    #maskzoltygreen = cv2.dilate(maskzoltygreen, kernel, iterations=4)

    #mask = cv2.bitwise_or(maskgreen, maskzoltygreen)

    (cnt, hierarchy) = cv2.findContours(
        maskgreen.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    green = len(cnt)

    lower_yellow = np.array([21, 192, 130])
    upper_yellow = np.array([33, 255, 255])

    maskyellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    #cv2.imwrite("green_image1.jpg", maskyellow)
    maskyellow = cv2.erode(maskyellow, kernel, iterations=1)
    cv2.imwrite("green_image2.jpg", maskyellow)
    maskyellow = cv2.morphologyEx(maskyellow, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("green_image2.jpg", maskgreen)
    maskyellow = cv2.dilate(maskyellow, kernel, iterations=2)

    (cnt, hierarchy) = cv2.findContours(
        maskyellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
    yellow = len(cnt)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img = cv2.GaussianBlur(img, (5, 5), 5)
    #lower_purple = np.array([138, 81, 18])
    #upper_purple = np.array([184, 213, 141])
    #lower_purple = np.array([123, 77, 38])
    #upper_purple = np.array([180, 255, 117])
    #H_scaled_lower = 310 / 2
    #S_scaled_lower = 30 * 255 / 100
    #V_scaled_lower = 15 * 255 / 100
    #H_scaled_upp = 340 / 2
    #S_scaled_upp = 80 * 255 / 100
    #V_scaled_upp = 80 * 255 / 100
   # lower_purple=cv2.cvtColor(lower_purple, cv2.COLOR_HSV2BGR)
   # upper_purple=cv2.cvtColor(upper_purple, cv2.COLOR_HSV2BGR)
    ##img_hsv=cv2.
    #maskpurple = cv2.inRange(img, lower_purple, upper_purple)
    #cv2.imwrite("green_image1.jpg", maskpurple)
    #maskpurple = cv2.erode(maskpurple, kernel, iterations=1)
    #cv2.imwrite("green_image2.jpg", maskpurple)
    #maskpurple = cv2.morphologyEx(maskpurple, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("green_image2.jpg", maskgreen)
    #maskpurple = cv2.dilate(maskpurple, kernel, iterations=4)
    #kernel = np.ones((7, 7), np.uint8)
    #maskpurple = cv2.morphologyEx(maskpurple, cv2.MORPH_OPEN, kernel)
    #maskpurple = cv2.morphologyEx(maskpurple, cv2.MORPH_OPEN, kernel)
    '''(cnt, hierarchy) = cv2.findContours(
        maskpurple.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(maskpurple, cv2.COLOR_BGR2RGB)
    cv2.drawContours(maskpurple, cnt, -1, (0, 255, 0), 4)
    purple = len(cnt)
    print('Numer of purple: ', len(cnt))
    '''
    #purple=0
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    down_width = 500
    down_height = 800
    down_points = (down_width, down_height)
    img = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
    img_hsv = cv2.resize(img_hsv, down_points, interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((5, 5), np.uint8)
    lower= np.array([135, 20, 0])
    upper = np.array([255, 255, 255])
    img_Kolor = cv2.inRange(img_hsv, lower, upper)
    #img_Kolor = cv2.erode(img_Kolor, kernel, iterations=1)
    img_Kolor = cv2.morphologyEx(img_Kolor, cv2.MORPH_OPEN, kernel)
    #img_Kolor = cv2.dilate(img_Kolor, kernel, iterations=2)
    img_Kolor = cv2.morphologyEx(img_Kolor, cv2.MORPH_CLOSE, kernel)
    fg = cv2.bitwise_or(img, img, mask=img_Kolor)
    mask = cv2.bitwise_not(img_Kolor)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bk)
    (cnt, hierarchy) = cv2.findContours(
        img_Kolor.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(img_Kolor, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 4)
    purple = len(cnt)-red
    #print('Numer of purple: ', len(cnt))


    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
