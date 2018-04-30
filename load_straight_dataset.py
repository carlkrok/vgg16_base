import glob
import os
import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

range_x=20
range_y=10

def get_index(angle):
    return int((angle+1)/0.02)


def load_dataset(camera_angle,lap, np_counter_array, aug_trans = True,aug_bright = True, aug_flip = True):
    
    LIMIT = 1000

# USE CAMERA ANGE [left, right, center], LAP [LEFT, RIGHT]

    if lap == "LEFT":
        csv_path = 'Datasets/driving_log_LEFT.csv'
    elif lap == "RIGHT":
        csv_path = 'Datasets/driving_log_RIGHT.csv'
    elif lap == "mond":
        csv_path = 'Datasets/driving_log_mond.csv'
    elif lap == "mond2":
        csv_path = 'Datasets/driving_log_mond2.csv'
    elif lap == "mond3":
        csv_path = 'Datasets/driving_log_mond3.csv'
    elif lap == "mond4":
        csv_path = 'Datasets/driving_log_mond4.csv'
    elif lap == "test":
        csv_path = 'Datasets/driving_log_test.csv'
    else:
        print("No dataset loaded")



    data_files = pd.read_csv(csv_path, index_col = False)
    data_files.columns = ['center','left','right','steer','throttle','break', 'speed']

    data_size = len(data_files)

    np_images = np.zeros((1, 64, 64, 3))
    np_steering = np.zeros(1)

    skip_count = 0
    
    image = cv2.imread(data_files[camera_angle][0].strip())
    counter = 0
    
    while image is None:
        counter += 1
        image = cv2.imread(data_files[camera_angle][counter].strip())
    
    steer = data_files['steer'][counter]
       
    #if camera_angle == "left" and steer > 0.8:
    #    continue
    #    skip_count += 1
    #elif camera_angle == "right" and steer < -0.8:
    #    continue
    #    skip_count += 1

    if camera_angle == "left":
        steer += 0.2
    elif camera_angle == "right":
        steer -= 0.2
    
    if aug_bright:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    shape = image.shape
    image = image[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
    image = image/255.-.5
    
    image = np.array(image)
        
    np_steering[0] = steer
    
    temp_img_array = np.zeros((1,64,64,3))
    temp_img_array[0] = image
    np_images[0] = image
    
    index = get_index(steer)
    np_counter_array[index] += 1

    for i_elem in range(counter, data_size):

        image = cv2.imread(data_files[camera_angle][i_elem].strip())

        if image is not None:

            steer = data_files['steer'][i_elem]
            if camera_angle == "left" and steer > 0.8:
                continue
                skip_count += 1
            elif camera_angle == "right" and steer < -0.8:
                continue
                skip_count += 1

            if camera_angle == "left":
                steer += 0.2
            elif camera_angle == "right":
                steer -= 0.2

            index = get_index(steer)

            if np_counter_array[index] < LIMIT and abs(steer) < 0.01:

                shape = image.shape
                image = image[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
                image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
                image = image/255.-.5

                image = np.array(image)

                temp_img_array = np.zeros((1,64,64,3))
                temp_img_array[0] = image

                np_images = np.concatenate((np_images, temp_img_array))
                np_steering = np.append(np_steering, 0.0)
                np_counter_array[index] += 1
                
             elif np_counter_array8[index] < LIMIT: 
                shape = image.shape
                image = image[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
                image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)
                image = image/255.-.5

                image = np.array(image)

                temp_img_array = np.zeros((1,64,64,3))
                temp_img_array[0] = image

                np_images = np.concatenate((np_images, temp_img_array))
                np_steering = np.append(np_steering, steer)
                np_counter_array[index] += 1

        else:
            skip_count += 1

    print("-----SKIPPED ", skip_count, " ITEMS")
    print("------- Number of images: ", len(np_images), " --- Number of angles: ", len(np_steering))

    return np_images, np_steering, np_counter_array;
