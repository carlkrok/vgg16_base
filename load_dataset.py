import glob
import os
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def load_dataset():

    csv_path = 'track4-counterclockwise.csv'

    data_files = pd.read_csv(csv_path, index_col = False)
    data_files['direction'] = pd.Series('s', index=data_files_s.index)
    data_files.columns = ['speed', 'steer', 'image', 'direction']

    rev_steer = np.array(data_files.steer,dtype=np.float32)
    steer_sm = rev_steer

    data_files['steer_sm'] = pd.Series(steer_sm, index=data_files.index)

    data_size = len(data_files)

    np_images = np.zeros((data_size, new_size_row, new_size_col, 3))
    np_steering = np.zeros(data_size)

    for i_elem in range(data_size):

        image = cv2.imread("/home/student/Desktop/Syndata/spurv_steering_angle/"+data['image'][i_elem].strip())

        if image is None:
            break

        shape = image.shape

        image = image[math.floor(shape[0]/4):, 0:shape[1]] #removed shape[0]-25 in row
        image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
        image = image/255.-.5
        image = np.array(image)

        steer = np.array([[data['steer_sm'][0]]])

        np_images[i_elem] = image
        np_steering[i_elem] = steer



    return np_images, np_steering;
