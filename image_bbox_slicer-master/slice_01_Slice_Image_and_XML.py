# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:13:37 2019

@author: paul
"""
#from importlib import reload
import os
import image_bbox_slicer as ibs
import pandas as pd

# Import file with relevant information for resizing
camp_info_file=r'resize_info.csv'

# Example image
# Folder with input images and XML files
im_src = r'slice_input'
an_src = im_src

# Folder to which rescaled images are written, for research purposes
im_res = r'slice_rescaled'
an_res = im_res

# Folder to which sliced images are written
im_out = r'slice_output'
an_out = im_out

# Max meter/pixel, to determine the resize factor before slicing the image
max_m_pixel=0.1
# Min meter/pixel, if an image has a resolution between min_m_pixel and max_m_pixel it will not be rescaled before slicing
min_m_pixel=0.1


os.makedirs(im_res,exist_ok=True)
os.makedirs(an_res,exist_ok=True)
os.makedirs(im_out,exist_ok=True)
os.makedirs(an_out,exist_ok=True)

# Read data with information on Range satellite image
data=pd.read_csv(camp_info_file)

# Initialize array with resize factors
resize_factor=[None]*len(data)

for i in range(len(data)):
	# Get altitude from Range parameter
    altitude=data.loc[i]['Range']
	# Transform altitude to meter_pixel
    meter_pixel=altitude*3.67*10**(-4)+5.97*10**(-5)
    if meter_pixel>max_m_pixel: # too many meters in a pixel, tent will be too small: scale up
        resize_factor[i]=meter_pixel/max_m_pixel
    elif meter_pixel<min_m_pixel: # too little meters in a pixel, causing a tent not to fit in one tile
        resize_factor[i]=meter_pixel/min_m_pixel
    else:
        resize_factor[i]=1
        
data['Resize_Factor']=resize_factor


# Resizing image
slicer = ibs.Slicer()
slicer.config_dirs(img_src=im_src, ann_src=an_src, 
                   img_dst=im_res, ann_dst=an_res)
slicer.resize_by_factor(resize_factor=data)

# Slicing image
slicer2 = ibs.Slicer()
slicer2.config_dirs(img_src=im_res, ann_src=an_res, 
                   img_dst=im_out, ann_dst=an_out)

slicer2.keep_partial_labels = True
slicer2.ignore_empty_tiles = False
slicer2.save_before_after_map = True
slicer2.slice_by_size(tile_size=(400,400))
slicer2.visualize_sliced_random(im_out)
