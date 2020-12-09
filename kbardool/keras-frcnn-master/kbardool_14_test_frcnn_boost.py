from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import re
import glob
from collections import defaultdict

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("--savePath", dest="save_path", help="Path to save data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_foldername", dest="config_foldername", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--threshold", dest="threshold", type="float", help="Threshold above which a detection is considered as a positive")
parser.add_option("--config_classifier", dest="config_classifier", help= "Location to read the metadata related to the training  classifier (generated when training).",
				default="config.pickle")
parser.add_option("--alpha_file", dest="alpha_file", help= "Location to read the alpha_t file.")
parser.add_option("--weights", dest="weights", help= "Weight number of the weight file, optional")
parser.add_option("--overlapThreshold", dest="overlapThreshold", type="float", help="If more overlap in detections than threshold, worst will be dismissed",default=0.1)


(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')

if not options.threshold:   # if filename is not given
	parser.error('Error: Detection threshold must be specified. Pass --threshold to command line')
	
if not options.save_path:   # if filename is not given
	parser.error('Error: Path to save data must be specified. Pass --savePath to command line')

config_dir = options.config_foldername
config_cls = options.config_classifier
alpha_file = options.alpha_file
overlap_threshold=options.overlapThreshold

if options.weights:
	we = int(options.weights)

img_path = options.test_path

threshold = float(options.threshold)
savePath=options.save_path
os.makedirs(savePath,exist_ok=True)

visualise = True
dict_RPN=defaultdict(list)

with open(alpha_file, 'rb') as f_in:
	alpha = pickle.load(f_in)
	
print(alpha)

for config_file in sorted(glob.glob(config_dir + '/*.pickle')):
	with open(config_file, 'rb') as f_in:
		C = pickle.load(f_in)
		
	if C.network == 'resnet50':
		import keras_frcnn.resnet as nn
	elif C.network == 'vgg':
		import keras_frcnn.vgg as nn
	
	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False

	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)
	
	class_mapping = {v: k for k, v in class_mapping.items()}
	print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	C.num_rois = int(options.num_rois)
	
	if C.network == 'resnet50':
		num_features = 1024
	elif C.network == 'vgg':
		num_features = 512
	
	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (num_features, None, None)
	else:
		input_shape_img = (None, None, 3)
		input_shape_features = (None, None, num_features)
	
	
	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)
	
	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)
	
	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)

	
	model_rpn = Model(img_input, rpn_layers)

	
	print('Loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	
	
	model_rpn.compile(optimizer='sgd', loss='mse')

	
	all_imgs = []
	
	classes = {}
	
	#P_cls_threshold after classification
	bbox_threshold = threshold
	
	#Get bounding boxes per boosted model
	for idx, img_name in enumerate(sorted(os.listdir(img_path))):
		if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
			continue
		SSID_slice=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}_[0-9]{6}",img_name).group(0)
		print(img_name)
		st = time.time()
		filepath = os.path.join(img_path,img_name)
	
		img = cv2.imread(filepath)
	
		X, ratio = format_img(img, C)
	
		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))
	
		# get the feature maps and output from the RPN
		[Y1, Y2, F] = model_rpn.predict(X)
		dict_RPN[img_name].append([Y1, Y2, F])
		print('Elapsed time = {}'.format(time.time() - st))


#Load the cls config		
with open(config_cls, 'rb') as f_in:
	C = pickle.load(f_in)
	
if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)
   
model_path_original=C.model_path
if options.weights:
	C.model_path=model_path_original[0:-5]+'_{0:0=2d}'.format(we)+model_path_original[-5:]

print('Loading weights from {}'.format(C.model_path))
model_classifier.load_weights(C.model_path, by_name=True)
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

#P_cls_threshold after classification
bbox_threshold = threshold


#Per image: combine bounding boxes to one list so NMS can be used.
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	SSID_slice=re.search("[A-Za-z]{2}_S[0-9]{3}_[0-9]{2,}_[0-9]{6}",img_name).group(0)
	f=open(savePath+'/'+'{}.txt'.format(SSID_slice),'w+')
#	print(SSID_slice)
	print(img_name)
	st = time.time()
	filepath = os.path.join(img_path,img_name)
	RPN_list=dict_RPN[img_name]
#	make lists to append all boxes and probs in
	R_x=[]
	p_x=[]
	alpha_list=[]
	tree=[]
	count_alpha=0
	for rpn_i in RPN_list:
		R_rpn, p_rpn = roi_helpers.rpn_to_roinoNMS(rpn_i[0], rpn_i[1], C, K.image_dim_ordering(), overlap_thresh=0.7)
		alpha_rpn=[alpha[count_alpha]]*len(p_rpn)
		tree_rpn=[count_alpha]*len(p_rpn)
		R_x.extend(R_rpn)
		p_x.extend(p_rpn)
		alpha_list.extend(alpha_rpn)
		tree.extend(tree_rpn)
		count_alpha+=1
	R,p=roi_helpers.non_max_suppression_fast_alpha(np.asarray(R_x), np.asarray(p_x), np.asarray(alpha_list), np.asarray(tree), overlap_thresh=0.7, max_boxes=300)
	
	
	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)

	if K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		F=RPN_list[0][2]
		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):
			# If white tent is larger than threshold: go on
			if (P_cls[0, ii, 0]) < bbox_threshold:
				continue
			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			
			
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])

			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])
		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=overlap_threshold) #If more overlap between two proposals than overlap_threshold, dismiss the worst
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
			f.write(str(key)+" "+str(new_probs[jk])+" "+str(real_x1)+" "+str(real_y1)+" " +str(real_x2)+" "+str(real_y2)+"\n")

	print('Elapsed time = {}'.format(time.time() - st))
	cv2.imwrite(savePath+'\\'+'{}.png'.format(SSID_slice),img)
	f.close()
print('All test images done, exiting')