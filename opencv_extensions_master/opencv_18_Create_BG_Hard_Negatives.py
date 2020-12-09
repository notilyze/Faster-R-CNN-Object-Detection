# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:19:58 2019

@author: paul
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:40:51 2019

@author: paul
"""

#from __future__ import division
import os
import cv2
import numpy as np
import sys
import time
import pandas as pd
import glob
import re
import shutil

# Random seed to make the chosen background images reproducible
random_seed=1

def get_img_name(img_path):
	return re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}(_[0-9]{6}){0,1}",img_path).group(0)

# Folder with both training image slices with objects in it, and the annotation .txt file
src = r'..\image_bbox_slicer-master\slice_output'
# Name of the .txt file with all training annotations, which must be in the src folder, from cv_02
training_ann_loc= "annotation_training.txt"

# Folder to save both the image slices with objects and the chosen image slices with bg label (optional, for analysis purposes)
dst_all=r'optional_output_from_opencv_18'

# Folder to save the new training .txt file with bg labels
dst_bg=r'..\image_bbox_slicer-master\slice_output'
# Name of the .txt file with image annotations and bg labels
training_ann_new_loc = "annotation_training_plus_bg.txt"

os.makedirs(dst_all,exist_ok=True)
os.makedirs(dst_bg,exist_ok=True)
#%%
training_path=os.path.join(src,training_ann_loc)
ann_file=pd.read_csv(training_path, delimiter=',', header=None, names=['img_path','xmin','ymin','xmax','ymax','class'])
ann_file['img_name']=ann_file['img_path'].apply(get_img_name)

files = [fn for fn in sorted(glob.glob(src + '/*.png'))
         if not os.path.basename(fn).endswith('annotated.png')]
count=0
empty_img_path_list=[None]*len(files)
min_x=[None]*len(files)
min_y=[None]*len(files)
max_x=[None]*len(files)
max_y=[None]*len(files)
class_name=[None]*len(files)
for img_path in files:
	img_name=get_img_name(img_path)

	selected=ann_file.loc[ann_file['img_name']==img_name][:]
	img = cv2.imread(img_path)

	if selected.empty==True:
		empty_img_path_list[count]=img_path
		min_x[count]=0
		min_y[count]=0
		max_x[count]=img.shape[1]
		max_y[count]=img.shape[0]
		class_name[count]='bg'
		
		count+=1
		continue
	

#%%
ann_bg=pd.DataFrame(
		{'img_path': empty_img_path_list[0:count],
   'xmin': min_x[0:count],
   'ymin': min_y[0:count],
   'xmax': max_x[0:count],
   'ymax': max_y[0:count],
   'class': class_name[0:count]
   })

ann_bg['img_name']=ann_bg['img_path'].apply(get_img_name)

#%%
# Take at most the same amount of image slices as bg slices as the amount of slices with objects
nr_bg=min((len(files)-len(ann_bg)),len(ann_bg))
chosen_bg=ann_bg.sample(n=nr_bg,replace=False,random_state=random_seed)


#%%

os.makedirs(dst_bg,exist_ok=True)
all_ann=pd.concat([ann_file,chosen_bg])
all_ann_sorted=all_ann.sort_values(by=['img_path'])
all_ann_sorted.to_csv(os.path.join(dst_bg,training_ann_new_loc),sep=',',columns=['img_path','xmin','ymin','xmax','ymax','class'],index=False,header=False)

#%% Make a file with both the new (background) images and the old set
for index, row in all_ann_sorted.iterrows():
	SSID_slice=row['img_name']
	shutil.copy(src+'\\'+SSID_slice+'.png',dst_all)
	