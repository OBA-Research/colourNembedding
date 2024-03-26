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
        
    def __getitem__(self, idx):
        #get data for one id value..pytorch will handle the batching for you!

        img_dir = self.paths[idx]
        label = self.labels[idx]
        img_id = self.image_id[idx]

        img = cv.imread(img_dir)
        img = img.transpose((2,0,1)) #channel must come first 
        img = torch.tensor(img, dtype = torch.float)
        label = torch.tensor(int(label), dtype = torch.int)
        return img/255.0, label, img_id