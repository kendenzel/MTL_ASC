from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import os
import torch
from PIL import Image # Replace by accimage when ready
from PIL.Image import FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM, ROTATE_90, ROTATE_180, ROTATE_270
from PIL.ImageEnhance import Color, Contrast, Brightness, Sharpness
from sklearn.preprocessing import MultiLabelBinarizer
from torch import np, from_numpy # Numpy like wrapper
from pathlib import Path

class loadDataSet(Dataset):


    def __init__(self, csv_path, img_path, transform=None):


        self.df = pd.read_csv(csv_path,names=['img_name','img_label'])

        # File Checking
        print("File Check ...")
        for index, row in self.df.iterrows():
            path = Path(os.path.join(img_path,row['img_name']))
            if path.exists():
                continue
            else:
                print(str(path) + " does not exist")


        self.img_path = img_path
        self.transform = transform

        self.labels = np.asarray(self.df.iloc[:,1])


    def X(self):
        return self.X

    def __getitem__(self, index):
        image_label = self.labels[index]
        image_name = self.df.iloc[index][0]
        img = Image.open(self.img_path + '/' + image_name)

            # Transform image to tensor
        if self.transform is not None:
            img_name = self.transform(img)

            # Return image and the label
        return (img_name, image_label)

    def __len__(self):
        return len(self.df.index)

    def getLabelEncoder(self):
        return self.mlb

    def getDF(self):
        return self.df