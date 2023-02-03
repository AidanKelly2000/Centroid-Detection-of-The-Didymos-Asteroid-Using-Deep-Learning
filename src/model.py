import torch.nn as nn
import torch.nn.functional as F

class AsteroidModel(nn.Module):
    def __init__(self):
        super(AsteroidModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=2, padding = 1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=2)
        self.fc1 = nn.Linear(1024, 2) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.175)
    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool1(x)
         x = F.relu(self.conv2(x))
         x = self.pool1(x)
         x = F.relu(self.conv3(x))
         x = self.pool1(x)
         x = F.relu(self.conv4(x))
         x = self.pool1(x)
         x = F.relu(self.conv5(x))
         x = F.relu(self.conv6(x))
         x = self.pool1(x)
         bs, _, _, _ = x.shape
         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
         x = self.dropout(x)
         out = self.fc1(x) 
         return out



"""
class AsteroidModel(nn.Module):
    def __init__(self):
        super(AsteroidModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=2)
        self.fc1 = nn.Linear(512, 2) 
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.175)
    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool1(x)
         x = F.relu(self.conv2(x))
         x = self.pool1(x)
         x = F.relu(self.conv3(x))
         x = self.pool1(x)
         x = F.relu(self.conv4(x))
         x = self.pool1(x)
         x = F.relu(self.conv5(x))
         x = self.pool1(x)
         bs, _, _, _ = x.shape
         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
         x = self.dropout(x)
         out = self.fc1(x) 
         return out"""
     
        
     
        
"""class AsteroidModel(nn.Module):
    def __init__(self):
        super(AsteroidModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256, 50) 
        self.fc2 = nn.Linear(50, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)
    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool(x)
         x = F.relu(self.conv2(x))
         x = self.pool(x)
         x = F.relu(self.conv3(x))
         x = self.pool(x)
         x = F.relu(self.conv4(x))
         x = self.pool(x)
         x = F.relu(self.fc1(x))
         x = self.pool(x)
         bs, _, _, _ = x.shape
         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
         x = self.dropout(x)
         out = self.fc2(x) 
         return out              """
        
     
"""
import torch.nn as nn
import torch.nn.functional as F
class AsteroidModel(nn.Module):
    def __init__(self):
        super(AsteroidModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=2)
        self.fc1 = nn.Linear(512, 2) 
        self.pool1 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.175)
    def forward(self, x):
         x = F.relu(self.conv1(x))
         x = self.pool1(x)
         x = F.relu(self.conv2(x))
         x = self.pool1(x)
         x = F.relu(self.conv3(x))
         x = self.pool1(x)
         x = F.relu(self.conv4(x))
         x = self.pool1(x)
         x = F.relu(self.conv5(x))
         x = self.pool1(x)
         x = F.relu(self.conv6(x))
         x = self.pool2(x)
         bs, _, _, _ = x.shape
         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
         x = self.dropout(x)
         out = self.fc1(x) 
         return out """
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        