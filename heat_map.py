import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import transform
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Activation
from vis.utils import utils
from vis.visualization import visualize_cam
from vis.visualization import visualize_saliency, overlay

def heat_map():

    FRAME_H = 64
    FRAME_W = 64

    model = load_model('trained_model_simulator.h5')
    img = utils.load_img('IMG_LEFT/center_2018_04_22_16_24_41_584.jpg')
    #img = utils.load_img('IMG_LEFT/center_2018_04_22_16_23_27_935.jpg')
    #img = utils.load_img('IMG_LEFT/center_2018_04_22_16_23_34_066.jpg')

    shape = img.shape
    img = img[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]

    target_size=(FRAME_H, FRAME_W)
    img = transform.resize(img, target_size, preserve_range=True).astype('uint8')

    plt.figure()
    plt.subplot()
    plt.imshow(img)
    plt.show()

    img_input = np.expand_dims(img_to_array(img), axis=0)

    pred = model.predict(img_input)[0][0]
    print('Predicted {}'.format(pred))

    titles = ['right steering', 'left steering', 'maintain steering']
    modifiers = [None, 'negate', 'small_values']


    for i, modifier in enumerate(modifiers):
        heatmap = visualize_saliency(model, layer_idx=-1, filter_indices=None, seed_input=img, grad_modifier=modifier) #, filter_indices=0
        plt.figure()
        plt.title(titles[i])
        # Overlay is used to alpha blend heatmap onto img.
        plt.imshow(overlay(img, heatmap, alpha=0.7))
        plt.show()

    for i, modifier in enumerate(modifiers):
        heatmap = visualize_cam(model, layer_idx=-1, filter_indices=None, seed_input=img, grad_modifier=modifier) #, filter_indices=0
        plt.figure()
        plt.title(titles[i])
        # Overlay is used to alpha blend heatmap onto img.
        plt.imshow(overlay(img, heatmap, alpha=0.7))
        #plt.imshow(overlay(img, heatmap, alpha=0.7))
        plt.show()


    return;
