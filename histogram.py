import numpy as np
import math
import matplotlib as plt
%matplotlib inline
import load_dataset_simulator

np_images, np_steering = load_dataset_simulator.load_dataset("right","RIGHT")

for dataset in ["LEFT", "RIGHT", "mond", "mond2", "mond3", "mond4"]:
    for camera_angle in ["center", "right", "left"]:

        if dataset == "RIGHT" and camera_angle == "right":
            break

        print("Currently loading dataset: ", dataset, ", angle: ", camera_angle, ".")
        np_images, new_np_steering = load_dataset_simulator.load_dataset(camera_angle,dataset)


        #len_tot_steering = len(new_np_steering) + len(np_steering)
        #np_images = np.concatenate((np_images, new_np_images))
        #np_steering = np.concatenate((np_steering, new_np_steering))
        np_steering = np.append((np_steering, new_np_steering))

        #print("Saving the model...")
        #save_load_model.save_model(model, "trained_model_simulator")
        print("Length of dataset so far: ", len(np_images), " images, ", len(np_steering), " steering angles.")


plt.hist(np_steering, bins=100)
plt.show()
