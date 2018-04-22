import glob
import os
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def load_dataset():

    csv_path = 'driving_log_RIGHT.csv'

    data_files = pd.read_csv(csv_path, index_col = False)
    data_files.columns = ['center','left','right','steer','throttle','break', 'speed']

    data_size = len(data_files)

    np_images = np.zeros((data_size, 64, 64, 3))
    np_steering = np.zeros(data_size)

    for i_elem in range(data_size):

        image = cv2.imread(data_files['center'][i_elem].strip())

        if image is not None:

            if i_elem%500 == 0:
                print("Image: ", data_files['center'][i_elem].strip(), " -- Steer: ", data_files['steer'][i_elem])

            shape = image.shape

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
            hsv[:,:,2] =  hsv[:,:,2] * ratio
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
            image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
            image = image/255.-.5

            steer = data_files['steer'][i_elem]

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                steer = -steer


            trans_x = range_x * (np.random.rand() - 0.5)
            trans_y = range_y * (np.random.rand() - 0.5)
            steer += trans_x * 0.002
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = image.shape[:2]
            image = cv2.warpAffine(image, trans_m, (width, height))

            image = np.array(image)

            np_images[i_elem] = image
            np_steering[i_elem] = steer

    return np_images, np_steering;


def load_dataset_left():

    csv_path = 'driving_log_RIGHT.csv'

    data_files = pd.read_csv(csv_path, index_col = False)
    data_files.columns = ['center','left','right','steer','throttle','break','speed']

    data_size = len(data_files)

    np_images = np.zeros((data_size, 64, 64, 3))
    np_steering = np.zeros(data_size)

    for i_elem in range(data_size):

        image = cv2.imread(data_files['left'][i_elem].strip())

        if image is not None:

            if i_elem%500 == 0:
                print("Image: ", data_files['left'][i_elem].strip(), " -- Steer: ", data_files['steer'][i_elem]+0.2)

            shape = image.shape

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
            hsv[:,:,2] =  hsv[:,:,2] * ratio
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
            image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
            image = image/255.-.5

            steer = data_files['steer'][i_elem]+0.2

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                steer = -steer

            image = np.array(image)

            np_images[i_elem] = image
            np_steering[i_elem] = steer

    return np_images, np_steering;


def load_dataset_right():

    csv_path = 'driving_log_RIGHT.csv'

    data_files = pd.read_csv(csv_path, index_col = False)
    data_files.columns = ['center','left','right','steer','throttle','break','speed']

    data_size = len(data_files)

    np_images = np.zeros((data_size, 64, 64, 3))
    np_steering = np.zeros(data_size)

    for i_elem in range(data_size):

        image = cv2.imread(data_files['right'][i_elem].strip())

        if image is not None:

            if i_elem%500 == 0:
                print("Image: ", data_files['right'][i_elem].strip(), " -- Steer: ", data_files['steer'][i_elem]-0.2)

            shape = image.shape

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
            hsv[:,:,2] =  hsv[:,:,2] * ratio
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
            image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
            image = image/255.-.5

            steer = data_files['steer'][i_elem]-0.2

            if np.random.rand() < 0.5:
                image = cv2.flip(image, 1)
                steer = -steer

            image = np.array(image)

            np_images[i_elem] = image
            np_steering[i_elem] = steer

    return np_images, np_steering;
