
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras as keras
from keras.preprocessing.image import ImageDataGenerator



yogadataset = "./yogadataset"

classes = sorted(os.listdir(yogadataset))
print(len(classes))
print(classes)

img_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load Data
train_ds = img_data_gen.flow_from_directory(yogadataset, target_size=(256,256), class_mode='binary', subset='training')
valid_ds = img_data_gen.flow_from_directory(yogadataset, target_size=(256,256), class_mode='binary', subset='validation')

