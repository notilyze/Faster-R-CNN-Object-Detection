# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:15:46 2019

@author: paul
"""

import os
import pandas as pd
import glob
import re
import cv2

def get_img_name(img_path):
	return re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}(_[0-9]{6}){0,1}",img_path).group(0)

# Src boxes and images are split, so you can use either the original image slices or the image slices with extra contrast
src_img=r'..\image_bbox_slicer-master\slice_output'
src_boxes=r'..\Object-Detection-Metrics-master\results\plot_boxes_white_tent_0.50.txt'
dst_img=r'results_evaluated_images'

os.makedirs(dst_img,exist_ok=True)

# Read all boxes in once for testing, change later to gt per file
bb=pd.read_csv(src_boxes, header=None, names=['SSID_slice','class','prob','xmin','ymin','xmax','ymax','confusion'])

count_empty=0
count_empty_correct=0
count_all=0

files = [fn for fn in sorted(glob.glob(src_img + '/*.png'))
         if not os.path.basename(fn).endswith('annotated.png')]

for img_path in files:
	count_all+=1
	img_name=get_img_name(img_path)
	img = cv2.imread(img_path)
	h,w,ch=img.shape
	selected=bb.loc[bb['SSID_slice']==img_name][:]
	
	# If no boxes or only FPs are found, the slice was originally empty
	if selected[selected["confusion"].str.contains('FP')==False].empty==True:
		count_empty+=1
	
	# Count correctly found empty slices (for analysis)
	if selected.empty==True:
		cv2.imwrite('{}\{}.png'.format(dst_img,img_name),img)
		count_empty_correct+=1
		continue
	
	for index, row in selected.iterrows():
		fn=0
		if row['confusion']=='TP':
			color=(0,255,127) #(green)
		elif (row['confusion']=='FP' or row['confusion']=='Double'):
			color=(0,165,255) #(orange)
		elif row['confusion']=='FN':
			color=(0,0,255) #(red)
			fn=1
			
		cv2.rectangle(img,(row['xmin'], row['ymin']), (row['xmax'], row['ymax']), color, 2)
		if fn==0:
			textLabel = '{}%'.format(int(row['prob']*100))
		elif fn==1:
			textLabel = 'FN'
		scale = 0.1 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
		fontScale = 0.5
		(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,fontScale,1)	  
		textOrg = (row['xmin'] + 2, row['ymin'] + retval[1] + 2)	

		cv2.rectangle(img, (row['xmin'], row['ymin']), (textOrg[0]+retval[0] + 2, textOrg[1] + 2), color, 2)
		cv2.rectangle(img, (row['xmin'], row['ymin']), (textOrg[0]+retval[0] + 2, textOrg[1] + 2), color, -1)		
		
		cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 0), 1)
	
	cv2.imwrite('{}\{}.png'.format(dst_img,img_name),img)
print("Total number of slices = {}, amount of slices that were correctly empty = {}/{}".format(count_all,count_empty_correct,count_empty))	