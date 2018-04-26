import vgg16
import load_dataset_simulator
import load_dataset_spurv
import save_load_model
#import heat_map
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import CSVLogger

import numpy as np
import math
import matplotlib.pyplot as plt



def main():

    print("Creating model...")
    vgg16.vgg16()

    #print("Loading model...")
    #model = load_model("vgg16_test.h5")
    #model = load_model("trained_model_simulator.h5")
    #model = load_model("model-010.h5")

    #csv_logger = CSVLogger('log_wednesday.csv', append=True, separator=';')
    #checkpoint = ModelCheckpoint('curr_best_model.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto') #Saved_models


    #print("Loading datasets...")
    np_images, np_steering = load_dataset_simulator.load_dataset("center","LEFT")

    #np_val_images, np_val_steering = load_dataset_simulator.load_dataset("center","test")
    print("Length of val images: ", len(np_val_images))
    #np.save("np_images",np_val_images)

    print("Length of val steer: ", len(np_val_steering))
    #np.save("np_steering",np_val_steering)


    for dataset in ["LEFT", "RIGHT", "mond", "mond2", "mond3", "mond4", "track1_rewind", "track2"]:
        for camera_angle in ["center", "right", "left"]:

            if dataset != "LEFT" and camera_angle != "center":
                #model = load_model("curr_best_model.h5")
                np_images_new, np_steering_new = load_dataset_simulator.load_dataset(camera_angle,dataset)

                #np_images = np.concatenate((np_images, np_images_new))
                np_steering = np.concatenate((np_steering, np_steering_new))



            #print("Currently loading dataset: ", dataset, ", angle: ", camera_angle, ".")
            #np_images, np_steering = load_dataset_simulator.load_dataset(camera_angle,dataset)

            #print("Training the model...")
            #history = model.fit(x=np_images, y=np_steering, epochs=50, batch_size=5, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images, np_val_steering))


            #len_tot_images = len(new_np_images) + len(np_images)
            #len_tot_steering = len(new_np_steering) + len(np_steering)
            #np_images = np.concatenate((np_images, new_np_images))
            #np_steering = np.concatenate((np_steering, new_np_steering))

            #print("Saving the model...")
            #save_load_model.save_model(model, "trained_model_simulator")
            #print("Length of dataset so far: ", len(np_images), " images, ", len(np_steering), " steering angles.")


    print("Saving the model...")

    #print("Length of images: ", len(np_images))
    #np.save("np_images",np_images)

    print("Length of steer: ", len(np_steering))
    np.save("np_steering",np_images)
    #save_load_model.save_model(model, "trained_model_wednesday")

    #plt.hist(np_steering, bins=100)
    #plt.show()

    #print("Creating heatmap...")
    #heat_map.heat_map()

    #with open('history1.txt', 'wb') as file_pi: #Saved_history/
    #    pickle.dump(history.history, file_pi)

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
