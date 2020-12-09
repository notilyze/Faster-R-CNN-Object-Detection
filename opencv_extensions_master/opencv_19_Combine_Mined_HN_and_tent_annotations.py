# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:29:33 2019

@author: paul
"""

import pandas as pd
import re

# Location plus name .txt file with tent annotations
src_tents_ann=r'..\image_bbox_slicer-master\slice_output\annotation_training.txt'

# Location plus name .txt file with hard negatives annotations
src_hn_ann=r'..\Object-Detection-Metrics-master\test\hard_negative_mining_white_tent_0.50.txt'

# Location to save the combined annotation .txt file -> This file can be used to 
# train on both labelled tents and the errors made during testing
# These errors are labelled as 'bg', but are smaller than the complete tiles from opencv_18
dst=r'..\image_bbox_slicer-master\slice_output\annotation_training_plus_hard_negatives.txt'


tents_ann=pd.read_csv(src_tents_ann, delimiter=',', header=None, names=['img_path','xmin','ymin','xmax','ymax','class'])
hn_ann=pd.read_csv(src_hn_ann, delimiter=',', header=None, names=['img_path','xmin','ymin','xmax','ymax','class'])

sample_img_path=tents_ann.img_path[0]
loc=sample_img_path[0:sample_img_path.rfind('/')+1]
hn_ann_new=hn_ann.replace({'folder/':loc},regex=True)
total_ann=pd.concat([tents_ann,hn_ann_new],ignore_index=True)
total_ann_sorted=total_ann.sort_values(['img_path'])

total_ann_sorted.to_csv(dst,index=False,header=False)
