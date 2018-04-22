import vgg16
import load_dataset
import save_load_model
import heat_map

def main():

    print("Creating model...")

    vgg16.vgg16()

    print("Loading model...")

    model = save_load_model.load_model("vgg16_test")

    print("Loading dataset...")

    np_images, np_steering = load_dataset.load_dataset()

    print("Training the model...")

    history = model.fit(np_images, np_steering, epochs=1, batch_size=32)

    print("Saving the model...")

    save_load_model.save_model(model, "trained_model")

    print("Creating heatmap...")

    heat_map.heat_map()

    print("Finished!")

    return 0;


if __name__== "__main__":
    main()
