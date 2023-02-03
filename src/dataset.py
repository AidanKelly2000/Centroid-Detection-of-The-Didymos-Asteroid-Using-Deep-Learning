import torch
import os
import cv2
import pandas as pd
import numpy as np
from skimage import io
import config
import utils
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

resize = 128

class AsteroidDataset(Dataset):
    
    def __init__(self, csv_file, root_dir):
       
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.images = []
        for i in tqdm(range(len(self.data))):
            img_name = os.path.join(self.root_dir,
                                    self.data.iloc[i, 0])
            image = io.imread(img_name)
            self.images.append(image)
    def __len__(self):
        
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index].reshape(1024, 1024)
        orig_w, orig_h = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (resize, resize))
        # again reshape to add grayscale channel format
        image = image.reshape(resize, resize, 1)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        # transpose for getting the channel size to index 0
        # get the keypoints
        keypoints = self.data.iloc[index][1:]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)
        # rescale keypoints according to image resize
        keypoints = keypoints * [resize / orig_w, resize / orig_h]
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float),
        }
    




# initialize the dataset - `AsteroidDataset()`
print('\n-------------- PREPARING DATA --------------\n')
train_data = AsteroidDataset(csv_file='C:\\Users\\ak234\\Personal-Python-Projects\\Centroid-Detection-of-The-Didymos-Asteroid-Using-Deep-Learning\\input\\asteroid com detection\\training\\training.csv' , root_dir='C:\\Users\\ak234\Personal-Python-Projects\\Centroid-Detection-of-The-Didymos-Asteroid-Using-Deep-Learning\\input\\asteroid com detection\\training\\training and validation images' )
valid_data = AsteroidDataset(csv_file='C:\\Users\\ak234\\Personal-Python-Projects\\Centroid-Detection-of-The-Didymos-Asteroid-Using-Deep-Learning\\input\\asteroid com detection\\validation\\validation.csv' , root_dir='C:\\Users\\ak234\\Personal-Python-Projects\\Centroid-Detection-of-The-Didymos-Asteroid-Using-Deep-Learning\\input\\asteroid com detection\\validation\\validation images' )
#valid_data = AsteroidDataset(valid_samples)
#print('\n-------------- DATA PREPRATION DONE --------------\n')
# prepare data loaders

train_loader = DataLoader(train_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(valid_data, 
                          batch_size=config.BATCH_SIZE, 
                          shuffle=False)

# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.dataset_keypoints_plot(valid_data)