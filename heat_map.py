#%matplotlib inline

import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import transform
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Activation
from vis.utils import utils
import cv2


FRAME_H = 64
FRAME_W = 64

pred_image = np.zeros((1, 64, 64, 3))

#model = load_model('trained_model_simulator.h5')
model = load_model('model-007.h5')
image = utils.load_img('Datasets/IMG_LEFT/center_2018_04_22_16_24_41_584.jpg') # Should be 0.3705882
#image = utils.load_img('Datasets/IMG_LEFT/center_2018_04_22_16_23_27_935.jpg')  # Should be -0.2117647
#image = utils.load_img('Datasets/IMG_LEFT/center_2018_04_22_16_23_34_066.jpg') # Should be -0.06470589

shape = image.shape
image_pred = image/255.-.5

image = np.array(image)
image_pred = np.array(image_pred)

image = image[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
image = cv2.resize(image,(64,64), interpolation=cv2.INTER_AREA)

image_pred = image_pred[int(math.floor(shape[0]/4)):shape[0]-25, 0:shape[1]]
image_pred = cv2.resize(image_pred,(64,64), interpolation=cv2.INTER_AREA)

plt.figure()
plt.subplot()
plt.imshow(image)
plt.show()

pred_image[0] = image_pred

pred = model.predict(pred_image)[0][0]
print('Predicted {}'.format(pred))

from vis.visualization import visualize_saliency, overlay

titles = ['right steering', 'left steering', 'maintain steering']
modifiers = [None, 'negate', 'small_values']


for i, modifier in enumerate(modifiers):
    heatmap = visualize_saliency(model, layer_idx=-1, filter_indices=None, seed_input=image_pred, grad_modifier=modifier) #, filter_indices=0
    plt.figure()
    plt.title(titles[i])
    # Overlay is used to alpha blend heatmap onto img.
    plt.imshow(overlay(image, heatmap, alpha=0.7))
    plt.show()

from vis.visualization import visualize_cam

for i, modifier in enumerate(modifiers):
    heatmap = visualize_cam(model, layer_idx=-1, filter_indices=None, seed_input=image_pred, grad_modifier=modifier) #, filter_indices=0
    plt.figure()
    plt.title(titles[i])
    # Overlay is used to alpha blend heatmap onto img.
    plt.imshow(overlay(image, heatmap, alpha=0.7))
    plt.show()
