# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:18:42 2020

@author: paul
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
import random
import shutil
from scipy import stats

#Data sample
src=r"..\image_bbox_slicer-master\slice_output"
dst_perc=r"..\image_bbox_slicer-master\slice_contrast"

os.makedirs(dst_perc,exist_ok=True)

files = [fn for fn in sorted(glob.glob(src + '/*.png'))]

for img_file in files:
	# Find SSID_slice name for file name
	SSID_slice=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}_[0-9]{6}",img_file).group(0)
	img1 = cv2.imread(os.path.join(src,SSID_slice+".png"))
	n_channels=img1.shape[2]
	# Initialize some arrays
	perc_img = np.zeros((img1.shape[0],img1.shape[1],n_channels),dtype = 'uint8')
	perc_2=[None]*n_channels
	perc_98=[None]*n_channels
	perc_img_corrected=np.empty((img1.shape[0],img1.shape[1],n_channels))
	perc_img_corrected[:]=np.nan
	# For each channel, calculate the 2 and 98 percentile values and stretch
	for k in range(n_channels):
		perc_2[k]=np.percentile(img1[:,:,k],2)
		perc_98[k]=np.percentile(img1[:,:,k],98)
		perc_img[:,:,k] = 255.0*(img1[:,:,k]-perc_2[k])/(perc_98[k]-perc_2[k])
		perc_img_corrected[:,:,k]=(np.where(img1[:,:,k]>perc_98[k],255,np.where(img1[:,:,k]<perc_2[k],0,perc_img[:,:,k])))
	cv2.imwrite(os.path.join(dst_perc,SSID_slice+".png"),perc_img_corrected)
