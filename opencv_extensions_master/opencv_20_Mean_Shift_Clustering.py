# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:59:28 2019

@author: paul
"""

import os
import pandas as pd
import csv
import cv2
import numpy as np
import re
from sklearn.cluster import MeanShift, estimate_bandwidth
import glob


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def get_img_name(img_path):
	return re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}(_[0-9]{6}){0,1}",img_path).group(0)

def name_to_path(src,img_name):
	return os.path.join(src,img_name+'.png')

def name_to_ann_path(src,img_name):
	return os.path.join(src,img_name+'.txt')

def translate_bb(x,trans):
	return x+trans

def write_empty_file(dst_folder,img_name):
	f=open(dst_folder+'\\'+img_name+'.txt','w+')
	f.close()

# Location of the mapper.csv file that has been created during slicing of the images
src_map=r'..\image_bbox_slicer-master\slice_output'

# Location of the bounding boxes (detected or ground truth), one .txt file with all bounding boxes in Faster R-CNN format
src_boxes=r'..\Object-Detection-Metrics-master\results\plot_boxes_white_tent_0.50.txt'

# Folder with detection text files (one per slice, direct result from e.g. kbardool_07_test_frcnn.py or from manual labels
# Make sure you use a folder with a text file for each slice (also for the ones without any detections, empty .txt file)
src_det= r'..\kbardool\keras-frcnn-master\images_results'

# Folder to save the new .txt files that only contains objects within the large clusters
dst_det_filtered=r'..\kbardool\keras-frcnn-master\detections_annotations_cluster'

# labels_eval=1 if you want to cluster annotation detection files with labels from evaluation (TP, FP, FN), 
# labels_eval=0 if you want to cluster annotation detection files without labels
labels_eval=1

os.makedirs(dst_det_filtered,exist_ok=True)
mapper=os.path.join(src_map,'mapper.csv')

with open(mapper) as csvfile:
	data = [row for row in csv.reader(csvfile)]
df = pd.DataFrame(data = data[1:])

df.rename(columns={0:'old_name'},inplace=True)
df.set_index('old_name',inplace=True)
nr_imgs=df.count(axis=1)

dict_imgs={}
appended_confusion=[]
appended_regr=[]

# Read all bounding boxes from that detection, those will be filtered out in the end
if labels_eval==1:
	bb=pd.read_csv(src_boxes , header=None, names=['SSID_slice','class','prob','xmin','ymin','xmax','ymax','GT'])
elif labels_eval==0:
	bb=pd.read_csv(src_boxes,index_col='Index')

count=0
for index, row in df.iloc[0:].iterrows():
	count+=1
	img_name=index
	base_name=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}",img_name).group(0)
	loc_img=os.path.join(os.path.dirname(src_map),index+'.png')
	img=cv2.imread(loc_img)
	h,w,ch=img.shape
	nr_img=nr_imgs[index]
	nr_width = int(np.ceil(w/400))
	nr_height=int(np.ceil(h/400))
	print(row[0:nr_img].values)
	ann_list=np.empty(nr_img,dtype=object)
	ann_files=np.empty(nr_img,dtype=object)
	for j in range(nr_img):
		ann_files[j]=name_to_ann_path(src_det,row[j+1])
		ann_list[j]=row[j+1] 
	ann_selected=bb[bb['SSID_slice'].isin(ann_list)].copy()
	if len(ann_selected)>0:
		
		img_matrix=ann_list.reshape(nr_height,nr_width)
		ann_selected.loc[:,'xmin_t']=-1
		ann_selected.loc[:,'ymin_t']=-1
		ann_selected.loc[:,'xmax_t']=-1
		ann_selected.loc[:,'ymax_t']=-1
		for i in range(nr_width):
			for j in range(nr_height):
				trans_x=i*400
				trans_y=j*400
				mask=(ann_selected['SSID_slice']==img_matrix[j,i])
				ann_selected.loc[mask,'xmin_t']=ann_selected['xmin']+trans_x
				ann_selected.loc[mask,'ymin_t']=ann_selected['ymin']+trans_y
				ann_selected.loc[mask,'xmax_t']=ann_selected['xmax']+trans_x
				ann_selected.loc[mask,'ymax_t']=ann_selected['ymax']+trans_y
		if labels_eval==1:
			mask=((ann_selected['GT'] == 'FP') | (ann_selected['GT'] == 'TP'))
			x=ann_selected.loc[mask].copy()
		elif labels_eval==0:
			x=ann_selected
			x['GT']=np.NaN
		x.loc[:,'xcenter']=(x['xmin_t']+x['xmax_t'])/2.0
		x.loc[:,'ycenter']=(x['ymin_t']+x['ymax_t'])/2.0
		x_feat=x.loc[:,'xcenter':'ycenter'].copy()
		y=pd.DataFrame(index=np.arange(len(x)))
		if labels_eval==1:
			y.loc[:,'GT']=x.loc[:,'GT']
		elif labels_eval==0:
			y['GT']=np.NaN
		# Clustering
		bandwidth = estimate_bandwidth(x_feat, quantile=0.2, n_samples=500)
		if bandwidth<0.0001:
			labels=np.zeros(len(y)).astype(int)
			centers=np.zeros((1,2))
			centers[0,0]=x['xcenter'].mean()
			centers[0,1]=x['ycenter'].mean()
		else:
			clustering=MeanShift(bandwidth=bandwidth,bin_seeding=False).fit(x_feat)
			labels=clustering.labels_
			centers=clustering.cluster_centers_
		
		y.loc[:,'Cluster_Labels']=labels
		y.loc[:,'SSID']=base_name
		x.loc[:,'dif_center_x']=x.loc[:,'xcenter']-centers[y.loc[:,'Cluster_Labels'],0]
		x.loc[:,'dif_center_y']=x.loc[:,'ycenter']-centers[y.loc[:,'Cluster_Labels'],1]
		x.loc[:,'Eucl_dist_center']=x.loc[:,'dif_center_x']**2+x.loc[:,'dif_center_y']**2
		x.loc[:,'Dist_center']=np.sqrt(x.loc[:,'dif_center_x']**2+x.loc[:,'dif_center_y']**2)
		
		confusion = y.groupby(['SSID','GT','Cluster_Labels']).size()
		appended_confusion.append(confusion)
		plot_df=pd.concat([x_feat,y],axis=1)
		groups=plot_df.groupby('Cluster_Labels')
	
		# Determine statistics to select correct cluster
		x.loc[:,'Cluster_Labels']=labels
		x.loc[:,'SSID']=base_name
		x.loc[:,'area']=(x.loc[:,'xmax']-x.loc[:,'xmin'])*(x.loc[:,'ymax']-x.loc[:,'ymin'])
		x.loc[:,'area_sq']=(np.sqrt(x.loc[:,'area']))
		
		if labels_eval==1:
			y_regr=x.groupby(['SSID','Cluster_Labels'])['GT'].apply(lambda x: (x=='TP').sum()).to_frame().rename(columns={'GT':'Count_TP'})
			x_regr=x.groupby(['SSID','Cluster_Labels']).agg({'GT': np.size, 'prob':np.mean, 'area':[np.mean,np.std],'xcenter':np.std,'ycenter':np.std,
						 'Eucl_dist_center':[np.mean,np.std], 'Dist_center':[np.mean,np.std] })
			x_regr['Percentage_detections']=x_regr[('GT','size')].apply(lambda x: x/int(x_regr.loc[:,'GT'].sum()))
		elif labels_eval==0:
			y_regr=x.groupby(['SSID','Cluster_Labels'])['GT'].apply(lambda x: (x=='NaN').sum()).to_frame().rename(columns={'GT':'Count_TP'})
			y_regr['Count_TP']=np.NaN
			x_regr=x.groupby(['SSID','Cluster_Labels']).agg({'prob':[np.size,np.mean], 'area':[np.mean,np.std],'xcenter':np.std,'ycenter':np.std,
						 'Eucl_dist_center':[np.mean,np.std], 'Dist_center':[np.mean,np.std] })
			x_regr['Percentage_detections']=x_regr[('prob','size')].apply(lambda x: x/int(x_regr.loc[:,('prob','size')].sum()))
		
		x_regr.columns=['Detections_amount','Prob_Detections_avg','Area_Detections_avg','Area_Detections_std','Xcenter_std','Ycenter_std',
						   'Eucl_dist_center_avg','Eucl_dist_center_std','Dist_center_avg','Dist_center_std','Detections_percentage']
		
		
		regr=pd.merge(y_regr, x_regr, left_index=True, right_index=True)
		selected_clusters=regr[regr['Detections_percentage']>0.25]
		filtered_detections=x.merge(selected_clusters, right_index=True, left_on=['SSID','Cluster_Labels']).loc[:,'SSID_slice':'GT']
		grouped=filtered_detections.groupby(filtered_detections.SSID_slice)
		for name, group in grouped:
			group.loc[:,'class':'ymax'].to_csv(r'{}/{}.txt'.format(dst_det_filtered, name), header=False, index=False, sep=' ')
	
# Make sure every image has a detection file, create empty files
SSID_slices_filtered=[get_img_name(file) for file in glob.glob(dst_det_filtered+'/*.txt')]
SSID_slices_original=[get_img_name(file) for file in glob.glob(src_det+'/*.txt')]
SSID_dif=list(set(SSID_slices_original)-set(SSID_slices_filtered))
[write_empty_file(dst_det_filtered,file) for file in SSID_dif]
