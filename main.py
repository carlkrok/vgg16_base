import vgg16
import load_dataset_simulator
import load_dataset_spurv
import save_load_model
import heat_map
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pickle
from keras.callbacks import CSVLogger



def main():

    print("Creating model...")
    vgg16.vgg16()

    print("Loading model...")
    model = load_model("vgg16_test.h5")
    #model = load_model("trained_model_simulator.h5")
    #model = load_model("model-010.h5")

    print("Loading datasets...")
    np_images, np_steering = load_dataset_simulator.load_dataset("right","RIGHT")
    np_val_images, np_val_steering = load_dataset_simulator.load_dataset("center","test")

    for dataset in ["LEFT"] #, "RIGHT", "mond", "mond2", "mond3", "mond4"]: #
        for camera_angle in ["center", "right", "left"]:

            if dataset == "RIGHT" && camera_angle == "right":
                break

            print("Currently loading dataset: ", dataset, ", angle: ", camera_angle, ".")
            new_np_images, new_np_steering = load_dataset_simulator.load_dataset(camera_angle,dataset)

            #len_tot_images = len(new_np_images) + len(np_images)
            #len_tot_steering = len(new_np_steering) + len(np_steering)
            np_images = np.append(np_images, new_np_images)
            np_steering = np.append(np_steering, new_np_steering)

            #print("Saving the model...")
            #save_load_model.save_model(model, "trained_model_simulator")
            print("Length of dataset so far: ", len(np_images), " images, ", len(np_steering), " steering angles.")


    print("Training the model...")
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto') #Saved_models
    history = model.fit(x=np_images, y=np_steering, epochs=50, batch_size=5, callbacks=[checkpoint, csv_logger], validation_data=(np_val_images, np_val_steering))

    print("Saving the model...")
    save_load_model.save_model(model, "trained_model_simulator")

    print("Creating heatmap...")
    heat_map.heat_map()

    #with open('history1.txt', 'wb') as file_pi: #Saved_history/
    #    pickle.dump(history.history, file_pi)

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
