# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:24:17 2019

@author: paul
"""

import glob
import xml.etree.ElementTree as ET
import re

# Source folder with label XML files in PASCAL VOC format, from labelImg.py (https://github.com/tzutalin/labelImg)
src=r'../image_bbox_slicer-master/slice_output'

# Absolute folder name where .png slices are found (can be changed if you want to use e.g. 
#images after contrast stretching for training, this location of images is used in the annotation_training.txt file )
src_images=r'../image_bbox_slicer-master\slice_output'

# Destination folder for .txt files per image slice, as we need those for evaluation of the model
dst=r'annotation_ground_truth'

# Destination .txt file to write combine all XML files in useful Faster R-CNN format
dst_total=src + "/annotation_training.txt"

# Initialize variables
names=[None]
xmin=str('')
ymin=str('')
xmax=str('')
ymax=str('')
className=str('')

# Create and open total file to write in
f_total = open(dst_total,"w+")

# Loop through src folder to get information from all XML files and transform it to Faster R-CNN format
for xml_file in sorted(glob.glob(src + '/*.xml')):
    SSID_slice=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}_[0-9]{6}",xml_file).group(0)
    fileName=src_images+'/'+SSID_slice+'.png'
    tree = ET.parse(xml_file)
    root = tree.getroot()

    f=open(dst+'\\{}.txt'.format(SSID_slice),'w+')
     
    for child in root:
        if child.tag=='object':
            className=child.findtext('name').replace(' ','_')
            for att in child:
                if att.tag=='bndbox':
                    xmin=att.findtext('xmin')
                    ymin=att.findtext('ymin')
                    xmax=att.findtext('xmax')
                    ymax=att.findtext('ymax')
            f_total.write(fileName + ',' + xmin + ',' + ymin + ',' + xmax + ',' + ymax + ',' + className + '\n')
            f.write(className + ' ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax +'\n')
    f.close()
f_total.close()
