# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:14:53 2022

@author: ak234
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import utils
import config
from model import AsteroidModel
from tqdm import tqdm

# image resize dimension
resize = 64

model = AsteroidModel()
# load the model checkpoint
checkpoint = torch.load(f"{config.OUTPUT_PATH}/model.pth") # loads the weights from the model path
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# read the test CSV file
csv_file = f"{config.ROOT_PATH}/test/test.csv"
data = pd.read_csv(csv_file)
pixel_col = data.Image
image_pixels = []
for i in tqdm(range(len(pixel_col))):
    img = pixel_col[i].split(' ')
    image_pixels.append(img)
# convert to NumPy array
images = np.array(image_pixels, dtype='float32')

images_list, outputs_list = [], []
for i in range(9):
    with torch.no_grad():
        image = images[i]
        image = image.reshape(64, 64, 1)
        image = cv2.resize(image, (resize, resize))
        image = image.reshape(resize, resize, 1)
        orig_image = image.copy()
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        image = image.unsqueeze(0)
        
        # forward pass through the model
        outputs = model(image)
        # append the current original image
        images_list.append(orig_image)
        # append the current outputs
        outputs_list.append(outputs)
utils.test_keypoints_plot(images_list, outputs_list)

""" We get the predicted keypoint at line15 and store them in outputs. After every forward pass, we are appending the image, 
and the outputs to the images_list and outputs_list respectively.

Finally, at line 22, we call the test_keypoint_plot() from utils that will plot the predicted keypoints on the images of the
 faces for us.

Execute the test.py script from the terminal/command prompt.

python test.py """