# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:40:51 2019

@author: paul
"""

from __future__ import division
import os
import cv2
import numpy as np
import sys
import time
import pandas as pd
import glob
import re
import shutil

def get_img_name(img_path):
	return re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}(_[0-9]{6}){0,1}",img_path).group(0)

# Image folder
src_img = r'..\image_bbox_slicer-master\slice_output'

# Ground Truth .txt files folder
src_gt=r'annotation_ground_truth'

# Path to folder to save images with ground truth boxes (if same as src_img, these images will be overwritten!)
savePath = r'ground_truth_images'

os.makedirs(savePath,exist_ok=True)

# Plot ground truth boxes in the images
for img_path in sorted(glob.glob(src_img + '/*.png')):
	img_name=get_img_name(img_path)
	img = cv2.imread(img_path)
	ann_file=pd.read_csv(os.path.join(src_gt,(img_name+'.txt')), delimiter=' ', header=None, names=['img_path','xmin','ymin','xmax','ymax','class'])
	selected=ann_file
	
	if selected.empty==True:
		cv2.imwrite('{}/{}.png'.format(savePath,img_name),img)
		continue
	
	for index, row in selected.iterrows():
		# RGB code for color of bounding boxes
		color=(255,255,255)
		# Plot rectangle
		cv2.rectangle(img,(row['xmin'], row['ymin']), (row['xmax'], row['ymax']), color, 2)
		# Plot text label
		textLabel = '{}'.format(row['class'])
		(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
		textOrg = (row['xmin'], row['ymin']-0)	
	# Save image		  
	cv2.imwrite('{}/{}.png'.format(savePath,img_name),img)

	