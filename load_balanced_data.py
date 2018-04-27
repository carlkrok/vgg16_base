import numpy as np
import math
import matplotlib
from importlib import reload
from matplotlib import pyplot as plt
%matplotlib inline
import load_dataset_simulator
reload(load_dataset_simulator)

np_steering = np.zeros(1)
np_counter_array = np.zeros(101)
print("----> np_counter_array, initially: ", np_counter_array)

np_images, np_steering, np_counter_array = load_dataset_simulator.load_dataset("right","RIGHT", np_counter_array)
print("----> np_counter_array, after right RIGHT: ", np_counter_array)


for dataset in ["LEFT", "RIGHT", "mond", "mond2", "mond3", "mond4"]:
    for camera_angle in ["center", "right", "left"]:

        if dataset == "RIGHT" and camera_angle == "right":
            continue

        print("Currently loading dataset: ", dataset, ", angle: ", camera_angle, ".")
        new_np_images, new_np_steering, np_counter_array = load_dataset_simulator.load_dataset(camera_angle,dataset, np_counter_array)
        
        print('-------> new_np_steering, after ', camera_angle, " ", dataset, ":", new_np_steering)
   
 
        np_steering = np.append(np_steering, new_np_steering)
        np_images = np.concatenate((np_images, new_np_images))

   
        print("Length of dataset so far: ", len(np_images), " images, ", len(np_steering), " steering angles.")
        print('-------> np_counter_array, after ', camera_angle, " ", dataset, ":", np_counter_array)
      
        plt.hist(np_steering, bins=100)
        plt.show()

plt.hist(np_steering, bins=100)
plt.show()
np.save("np_steering_angles_log.npy", np_steering)
np.save("np_images_balanced.npy", np_images)
