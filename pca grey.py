# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:31:51 2021

@author: shwet
"""

#import cv2
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from skimage import io ,img_as_ubyte

img = io.imread('image00067.jpeg')
io.imshow(img)

pca = PCA(256)

img_transformed = pca.fit_transform(img)
img_inverted = pca.inverse_transform(img_transformed)

#img_compressed = (np.dstack((blue_inverted, green_inverted, red_inverted))).astype(np.uint8)
#image = img_as_ubyte(img_transformed)
plt.imshow(img_transformed, cmap = 'gray')
img_inverted.shape
img.shape

