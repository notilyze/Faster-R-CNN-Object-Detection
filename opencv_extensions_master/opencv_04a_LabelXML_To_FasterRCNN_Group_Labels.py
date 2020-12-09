# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:24:17 2019

@author: paul
"""

import os
import glob
import xml.etree.ElementTree as ET
import re

# Source folder with label XML files in PASCAL VOC format, from labelImg.py (https://github.com/tzutalin/labelImg)
src=r'../image_bbox_slicer-master/slice_output'

# Absolute folder name where .png slices are found (can be changed if you want to use e.g. 
#images after contrast stretching for training, this location of images is used in the annotation_training.txt file )
src_images=r'..\image_bbox_slicer-master\slice_output'

# Destination folder for .txt files per image slice, as we need those for evaluation of the model
dst=r'annotation_ground_truth_grouped'

# Destination .txt file to write combine all XML files in useful Faster R-CNN format
dst_total=src + "/annotation_training_GROUP.txt"

# Initialize variables
names=[None]
className=str('')

# Create and open file to write in
os.makedirs(dst,exist_ok=True)
f_total= open(dst_total,"w+")

# Loop through src folder to get information from all XML files and transform it to Faster R-CNN format
for xml_file in sorted(glob.glob(src + '/*.xml')):
	SSID_slice=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}_[0-9]{6}",xml_file).group(0)
	fileName=src_images+'/'+SSID_slice+'.png'
	tree = ET.parse(xml_file)
	root = tree.getroot()
	f=open(dst+'\\{}.txt'.format(SSID_slice),'w+')
	
	# Slices of size 400x400 determine the initial values of xmin_total and ymin_total
	xmin_total=401
	ymin_total=401
	xmax_total=-1
	ymax_total=-1
	count=0
	found=0

	for child in root:
		if child.tag=='object':
			# Replace accidental "white tent" with "white_tent"
			className=child.findtext('name').replace(' ','_')
			if className =='white_tent':
				for att in child:
					# Find bounding box and check if it is a new border of the group of tents in that image slice
					if att.tag=='bndbox':
						xmin=int(att.findtext('xmin'))
						ymin=int(att.findtext('ymin'))
						xmax=int(att.findtext('xmax'))
						ymax=int(att.findtext('ymax'))
						xmin_total=min(xmin_total,xmin)
						ymin_total=min(ymin_total,ymin)
						xmax_total=max(xmax_total,xmax)
						ymax_total=max(ymax_total,ymax)
				if xmin==0 or ymin==0 or xmax==400 or ymax==400:
					if xmax-xmin>30 and ymax-ymin>30:
						found=1
						count+=1
				else:
					found=1
					count+=1
	if found==1:
		f.write('white_tent_' + str(count) + ' ' + str(xmin_total) + ' ' + str(ymin_total) + ' ' + str(xmax_total) + ' ' + str(ymax_total) + '\n')
		f_total.write(fileName + ',' + str(xmin_total) + ',' + str(ymin_total) + ',' + str(xmax_total) + ',' + str(ymax_total) + ',' + 'white_tent_' + str(count) + '\n')
	f.close()
f_total.close()
