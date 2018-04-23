import vgg16
import load_dataset_left
import load_dataset_right
import load_dataset_spurv
import save_load_model
import heat_map
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import pickle

def main():

    #print("Creating model...")
    #vgg16.vgg16()

    print("Loading model...")
    #model = load_model("vgg16_test.h5")
    model = load_model("trained_model_simulator.h5")

    checkpoint = ModelCheckpoint('Saved_models/model-{epoch:03d}.h5', monitor='val_loss',verbose=0,save_best_only=True, mode='auto')

    print("Loading dataset LEFT - center...")
    np_images, np_steering = load_dataset_simulator.load_dataset("center","LEFT")

    print("Training the model...")
    history = model.fit(np_images, np_steering, epochs=5, batch_size=5, callbacks=[checkpoint])
    with open('Saved_history/', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    print("Loading dataset LEFT - left...")
    np_images, np_steering = load_dataset_simulator.load_dataset("left","LEFT")

    print("Training the model...")
    history = model.fit(np_images, np_steering, epochs=5, batch_size=5, callbacks=[checkpoint])
    with open('Saved_history/', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    print("Loading dataset LEFT - right...")
    np_images, np_steering = load_dataset_simulator.load_dataset("right","LEFT")

    print("Training the model...")
    history = model.fit(np_images, np_steering, epochs=5, batch_size=5, callbacks=[checkpoint])

    print("Loading dataset RIGHT - center...")
    np_images, np_steering = load_dataset_simulator.load_dataset("center","RIGHT")

    print("Training the model...")
    history = model.fit(np_images, np_steering, epochs=5, batch_size=5, callbacks=[checkpoint])

    print("Loading dataset RIGHT - left...")
    np_images, np_steering = load_dataset_simulator.load_dataset("left","RIGHT")

    print("Training the model...")
    history = model.fit(np_images, np_steering, epochs=5, batch_size=5, callbacks=[checkpoint])

    print("Loading dataset RIGHT - right...")
    np_images, np_steering = load_dataset_simulator.load_dataset("right","RIGHT")

    print("Training the model...")
    history = model.fit(np_images, np_steering, epochs=5, batch_size=5, callbacks=[checkpoint])
    with open('Saved_history/', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    print("Saving the model...")
    save_load_model.save_model(model, "trained_model_simulator")

    #print("Creating heatmap...")
    #heat_map.heat_map()

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
