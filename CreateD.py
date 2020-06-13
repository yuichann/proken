import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO 
import os
import glob
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
spring,summer,autumn,winter = [],[],[],[]


def import_img(filename):
    par_id = os.listdir(filename)
    for count in par_id:
        dir_label = os.path.join(filename, count)
        for file_name in os.listdir(dir_label):
            img_name = os.path.join(dir_label,file_name)
            print(img_name)
            if img_name.find('spring') != -1:
                spring.append(img_name)
            elif img_name.find('summer') != -1:
                summer.append(img_name)
            elif img_name.find('fall') != -1:
                autumn.append(img_name)
            elif img_name.find('autumn') != -1:
                autumn.append(img_name)
            elif img_name.find('winter') != -1:
                winter.append(img_name)

    
            # img = load_img(sa, target_size=(hw["height"], hw["width"])) 
            # array = img_to_array(img) / 255
            
            
            
    

def separate_img(filename):
    return 0


filename = '../fashiondata_tng/photos'
import_img(filename)
par_id = os.listdir(filename)