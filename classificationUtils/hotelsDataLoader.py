import numpy as np
import torch
from torch.utils.data import Dataset
import cv2 as cv

class HOTELS(Dataset):
    def __init__(self, paths_labels_and_ids, unique_labels):
        #get all necessary inputs like train directories and labels
        self.paths = [i[0] for i in paths_labels_and_ids]
        self.labels = [i[1] for i in paths_labels_and_ids]
        self.image_id = [i[2] for i in paths_labels_and_ids]
        self.unique_labels = np.asarray(unique_labels)


        
    def __len__(self,):
        #return len of the dataset
        return len(self.paths)
    
    def get_one_hot_encoding(self, cat):
        one_hot = np.asarray(cat == self.unique_labels)
        return one_hot
        
    def __getitem__(self, idx):
        #get data for one id value..pytorch will handle the batching for you!

        img_dir = self.paths[idx]
        label = self.labels[idx]
        img_id = self.image_id[idx]

        img = cv.imread(img_dir)
        # img = cv.resize(img,(256,256))
        one_hot = self.get_one_hot_encoding(label)

        img = img.transpose((2,0,1)) #channel must come first 
        img = torch.tensor(img, dtype = torch.float)
        one_hot = torch.tensor(one_hot, dtype = torch.float)
        return img/255.0, one_hot, img_id