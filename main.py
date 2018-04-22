import vgg16
import load_dataset_left
import load_dataset_right
import load_dataset_spurv
import save_load_model
import heat_map
from keras.models import load_model

def main():

    #print("Creating model...")

    #vgg16.vgg16()

    print("Loading model...")

    #model = load_model("vgg16_test.h5")
    model = load_model("trained_model_simulator.h5")

    print("Loading dataset left...")

    np_images, np_steering = load_dataset_left.load_dataset()

    print("Training the model...")

    history = model.fit(np_images, np_steering, epochs=5, batch_size=7)

    print("Loading dataset right...")

    np_images, np_steering = load_dataset_right.load_dataset()

    print("Training the model...")

    history = model.fit(np_images, np_steering, epochs=5, batch_size=7)

    print("Saving the model...")

    save_load_model.save_model(model, "trained_model_simulator")

    #print("Creating heatmap...")

    #heat_map.heat_map()

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
