from keras.models import load_model

def save_model(model, model_name):

    model.save(model_name+".h5")

    print("Saved model to disk")

    return;
