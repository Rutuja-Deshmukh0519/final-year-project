# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 15:03:59 2021

@author: shwet
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io

img = io.imread('n01440764_1775.jpeg')
plt.imshow(img)

# Splitting the image in R,G,B arrays.
 
blue,green,red = cv2.split(img) 

#initialize PCA with first 20 principal components
pca = PCA(350)

#Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)

#Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)

#Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)

img_compressed = (np.dstack((blue_inverted, green_inverted, red_inverted))).astype(np.uint8)


plt.imshow(img_compressed)

# PC_values = np.arange(pca.n_components_) + 1
# plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Variance Explained')
# plt.show()















