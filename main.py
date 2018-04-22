import vgg16
import load_dataset
import save_load_model
import heat_map
from keras.models import load_model

def main():

    print("Creating model...")

    vgg16.vgg16()

    print("Loading model...")

    model = load_model("vgg16_test.h5")

    print("Loading dataset...")

    np_images, np_steering = load_dataset.load_dataset()

    print("Training the model...")

    history = model.fit(np_images, np_steering, epochs=5, batch_size=10)

    print("Saving the model...")

    save_load_model.save_model(model, "trained_model")

    #print("Creating heatmap...")

    #heat_map.heat_map()

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
