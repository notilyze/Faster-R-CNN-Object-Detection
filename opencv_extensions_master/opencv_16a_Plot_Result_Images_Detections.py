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
import sys
import importlib

# Add path of Object-Detection-Metrics lib, so we can use those functions
sys.path.insert(1, r'Object-Detection-Metrics-master\lib')
sys.path.append('Object-Detection-Metrics-master\lib')

from BoundingBoxes import BoundingBoxes
from BoundingBox import BoundingBox
from BoundingBox import *
from BoundingBoxes import *
from utils import *
from utils import BBFormat

# Locations of respectively 1. image slices, 2. folder with detections (.txt) and 3. destination of the plots. Must be absolute paths!
src_img=r'..\image_bbox_slicer-master\slice_output'
detFolder=r'..\kbardool\keras-frcnn-master\images_results'
dst_img=r'results_images'

os.makedirs(dst_img,exist_ok=True)

def get_img_name(img_path):
	return re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}(_[0-9]{6}){0,1}",img_path).group(0)

# Function from Object-Detection-Metrics project, written by rafael padilla
def getBoundingBoxes(directory,
					 isGT,
					 bbFormat,
					 coordType,
					 allBoundingBoxes=None,
					 allClasses=None,
					 imgSize=(0, 0)):
	"""Read txt files containing bounding boxes (ground truth and detections)."""
	if allBoundingBoxes is None:
		allBoundingBoxes = BoundingBoxes()
	if allClasses is None:
		allClasses = []
	# Read ground truths
	os.chdir(directory)
	files = glob.glob("*.txt")
	files.sort()
	# Read GT detections from txt file
	# Each line of the files in the groundtruths folder represents a ground truth bounding box
	# (bounding boxes that a detector should detect)
	# Each value of each line is  "class_id, x, y, width, height" respectively
	# Class_id represents the class of the bounding box
	# x, y represents the most top-left coordinates of the bounding box
	# x2, y2 represents the most bottom-right coordinates of the bounding box
	for f in files:
		nameOfImage = f.replace(".txt", "")
		fh1 = open(f, "r")
		for line in fh1:
			line = line.replace("\n", "")
			if line.replace(' ', '') == '':
				continue
			splitLine = line.split(" ")
			if isGT:
				# idClass = int(splitLine[0]) #class
				idClass = (splitLine[0])  # class
				x = float(splitLine[1])
				y = float(splitLine[2])
				w = float(splitLine[3])
				h = float(splitLine[4])
				bb = BoundingBox(
					nameOfImage,
					idClass,
					x,
					y,
					w,
					h,
					coordType,
					imgSize,
					BBType.GroundTruth,
					format=bbFormat)
			else:
				# idClass = int(splitLine[0]) #class
				idClass = (splitLine[0])  # class
				confidence = float(splitLine[1])
				x = float(splitLine[2])
				y = float(splitLine[3])
				w = float(splitLine[4])
				h = float(splitLine[5])
				bb = BoundingBox(
					nameOfImage,
					idClass,
					x,
					y,
					w,
					h,
					coordType,
					imgSize,
					BBType.Detected,
					confidence,
					format=bbFormat)
			allBoundingBoxes.addBoundingBox(bb)
			if idClass not in allClasses:
				allClasses.append(idClass)
		fh1.close()
	return allBoundingBoxes, allClasses

detFormat = 'xyrb'
detCoordType= 'abs'

allBoundingBoxes, allClasses = getBoundingBoxes(detFolder, False, detFormat, detCoordType)
detections=[]
classes=[]
groundTruths=[] #if correctly loaded, this list should still be empty after this script has been run
for bb in allBoundingBoxes.getBoundingBoxes():
	# [imageName, class, confidence, (bb coordinates X1Y1X2Y2)]
	if bb.getBBType() == BBType.GroundTruth:
		groundTruths.append([
			bb.getImageName(),
			bb.getClassId(), 1,
			bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
		])
	else:
		detections.append([
			bb.getImageName(),
			bb.getClassId(),
			bb.getConfidence(),
			bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
		])
	# get class
	if bb.getClassId() not in classes:
		classes.append(bb.getClassId())
classes = sorted(classes)


bb=pd.DataFrame.from_records(detections, columns=['SSID_slice','class','prob','coords'])
bb[['xmin','ymin','xmax','ymax']] = pd.DataFrame(bb['coords'].tolist(),index=bb.index)
bb[['xmin','ymin','xmax','ymax']] = bb[['xmin','ymin','xmax','ymax']].astype(int)
bb=bb.drop('coords',1)
bb.to_csv((dst_img+'\detections_all_{}.csv'.format(classes[0])),index=True,index_label='Index')
#%% Create and write .png files

count_all=0

files = [fn for fn in sorted(glob.glob(src_img + '/*.png'))
         if not os.path.basename(fn).endswith('annotated.png')]

for img_path in files:
	count_all+=1
	img_name=get_img_name(img_path)
	img = cv2.imread(img_path)
	h,w,ch=img.shape
	selected=bb.loc[bb['SSID_slice']==img_name][:]
	
	if selected.empty==True:
		cv2.imwrite('{}\{}.png'.format(dst_img,img_name),img)
		continue
	
	color=(255,255,255)
	
	for index, row in selected.iterrows():
		cv2.rectangle(img,(int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), color, 2)
	
		textLabel = '{}%'.format(int(row['prob']*100))
		scale = 0.1 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
		fontScale = 0.5
		(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,fontScale,1)		  
		textOrg = (row['xmin'] + 2, row['ymin'] + retval[1] + 2)	
    
		cv2.rectangle(img, (row['xmin'], row['ymin']), (textOrg[0]+retval[0] + 2, textOrg[1] + 2), color, 2)
		cv2.rectangle(img, (row['xmin'], row['ymin']), (textOrg[0]+retval[0] + 2, textOrg[1] + 2), color, -1)		
		
		cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, fontScale, (0, 0, 0), 1)
	
	
	cv2.imwrite('{}\{}.png'.format(dst_img,img_name),img)
print("Total number of slices done = {}".format(count_all))	