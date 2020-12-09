# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:08:34 2019

@author: paul
"""
import os
import pandas as pd
import csv
import cv2
import numpy as np
import re


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def get_img_name(img_path):
	return re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}(_[0-9]{6}){0,1}",img_path).group(0)

def name_to_path(src,img_name):
	return os.path.join(src,img_name+'.png')



# Folder in which the mapper.csv file is located that has been created during slicing of the images
src_map=r'..\image_bbox_slicer-master\slice_output'
# Folder with image slices to combine (can come from either opencv_16 or opencv_16a)
src_tiles= r'results_evaluated_images'
# Folder to save combined image
dst_img=r'detections_slices_combined'
os.makedirs(dst_img,exist_ok=True)

mapper=os.path.join(src_map,'mapper.csv')

with open(mapper) as csvfile:
	data = [row for row in csv.reader(csvfile)]
df = pd.DataFrame(data = data[1:])

df.rename(columns={0:'old_name'},inplace=True)
df.set_index('old_name',inplace=True)
nr_imgs=df.count(axis=1)

dict_imgs={}
for index, row in df.iterrows():
	img_name=index
	base_name=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}",img_name).group(0)
	loc_img=os.path.join(os.path.dirname(src_map),index+'.png')
	img=cv2.imread(loc_img)
	h,w,ch=img.shape
	nr_img=nr_imgs[index]
	nr_width = int(np.ceil(w/400))
	nr_height=int(np.ceil(h/400))
	print(row[0:nr_img].values)
	img_list=np.empty(nr_img,dtype=object)
	for j in range(nr_img):
		img_list[j]=cv2.imread(name_to_path(src_tiles,row[j+1]))
	img_matrix=img_list.reshape(nr_height,nr_width)
	total_image = concat_tile(img_matrix)
	cv2.imwrite(os.path.join(dst_img,base_name+'.png'),total_image)