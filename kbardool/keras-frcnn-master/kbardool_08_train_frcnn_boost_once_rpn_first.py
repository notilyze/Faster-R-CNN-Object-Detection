from __future__ import division

import numpy as np
import random
import tensorflow as tf
import pprint
import sys
import time
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config 
from keras_frcnn import data_generators_boost as data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers_boost as roi_helpers
from keras.utils import generic_utils
from collections import defaultdict
import operator

def write_pickle(data, outfile):
        f = open(outfile, "wb")
        pickle.dump(data, f)
        f.close()
		
def read_pickle(filename):
        f = open(filename,'rb')
        data = pickle.load(f)
        f.close()
        return data

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
				default="pascal_voc")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=False)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).",
				  action="store_true", default=False)
parser.add_option("--epoch_length", type="int", dest="epoch_length", help="Epoch length. Only 1 epoch is run, so this is the only parameter to change the amount of training per boost tree", default=1000)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='./model_frcnn.hdf5')
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")
parser.add_option("--output_boost_path", dest="output_boost_path", help="Output path for boost importance.", default='./alpha_t.file')

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
	from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)
C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)
C.output_boost = options.output_boost_path

if options.network == 'vgg':
	C.network = 'vgg'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
else:
	print('Not a valid model')
	raise ValueError


# check if weight path was passed via command line
if options.input_weight_path:
	C.base_net_weights = options.input_weight_path
else:
	# set the path to weights based on backend and model
	C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']

print('Num train samples {}'.format(len(train_imgs)))

num_train_imgs=len(train_imgs)
#Create dictionary for train_imgs names for fast finding of the picked train images (train_dict)
train_dict = {}

#Also create a dictionary to keep track of boosting weight per image
weight_dict={}
for i in range(num_train_imgs):
	train_dict[train_imgs[i]['filepath']]=train_imgs[i]
	weight_dict[train_imgs[i]['filepath']]=1.0/num_train_imgs


epoch_length = int(options.epoch_length)

data_gen_train = data_generators.get_anchor_gt(train_dict, weight_dict, epoch_length, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')


if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
else:
	input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
print(img_input)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
classifier_old=nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=2, trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)
model_classifier_old = Model([img_input, roi_input], classifier_old)																	

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
	print('loading weights from {}'.format(C.base_net_weights))
	model_rpn.load_weights(C.base_net_weights, by_name=True)
	model_classifier_old.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')
weights_transfer=[None]*len(model_classifier_old.layers[:])
print('Number of layers: '+str(len(model_classifier_old.layers[:])))
for i in range(len(model_classifier_old.layers[:])-2):
	weights_transfer[i]=model_classifier_old.layers[i].get_weights()
	model_classifier.layers[i].set_weights(weights_transfer[i])

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

# Don't change num_epochs; each boosting tree is one epoch. To train one tree longer, use --epoch_length instead
num_epochs = 1
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True
error_log_dict=defaultdict(list) #dictionary with all losses per image in one epoch
error_dict={} #dictionary with only the latest error per training img in that epoch
weight_update_dict={} #dictionary with weights that need to be updated in that epoch
alpha_t=np.zeros(num_epochs)
for epoch_num in range(num_epochs):
	error_dict.clear()
	error_log_dict.clear()
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:
		try:

			if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
				mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
				rpn_accuracy_rpn_monitor = []
				print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
				if mean_overlapping_bboxes == 0:
					print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

			X, Y, img_data = next(data_gen_train)
			print(img_data['filepath'])
			loss_rpn = model_rpn.train_on_batch(X, Y)

			P_rpn = model_rpn.predict_on_batch(X)
			error_log_dict[img_data['filepath']].append(loss_rpn+P_rpn)
			error_dict[img_data['filepath']]=loss_rpn[1]+loss_rpn[2]									   
			R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)[0]
			# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

			if X2 is None:
				rpn_accuracy_rpn_monitor.append(0)
				rpn_accuracy_for_epoch.append(0)
				continue

			neg_samples = np.where(Y1[0, :, -1] == 1)
			pos_samples = np.where(Y1[0, :, -1] == 0)

			if len(neg_samples) > 0:
				neg_samples = neg_samples[0]
			else:
				neg_samples = []

			if len(pos_samples) > 0:
				pos_samples = pos_samples[0]
			else:
				pos_samples = []
			
			rpn_accuracy_rpn_monitor.append(len(pos_samples))
			rpn_accuracy_for_epoch.append((len(pos_samples)))


			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]


			progbar.update(iter_num+1, [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
									  ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

			iter_num += 1
			
			if iter_num == epoch_length:
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_class_cls = np.mean(losses[:, 2])
				loss_class_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []
				
				alpha_t[epoch_num]=1/(loss_rpn_cls+loss_rpn_regr+0.3)
				sum_exponents=0
				sum_errors=0
				weight_update_dict.clear()
				for key in error_dict:
					error_prev=weight_dict[key]
					weight_update_dict[key]=error_prev*np.exp(alpha_t[epoch_num]*error_dict[key])
					sum_exponents+=weight_update_dict[key]
					sum_errors+=error_prev
					
				for key in weight_update_dict:
					weight_dict[key]=weight_update_dict[key]/sum_exponents*sum_errors
				
				max_weight=max(weight_dict.items(),key=operator.itemgetter(1))
				min_weight=min(weight_dict.items(),key=operator.itemgetter(1))
				
				if C.verbose:
					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))
					print('Max weight Image, Key:Value= {}:{}'.format(max_weight[0],max_weight[1]))
					print('Min weight Image, Key:Value= {}:{}'.format(min_weight[0],min_weight[1]))																					

				curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
				iter_num = 0
				start_time = time.time()
				
				model_all.save_weights(C.model_path)

				break

		except Exception as e:
			print('Exception: {}'.format(e))
			tb = sys.exc_info()[2]
			print(tb.tb_lineno)						 
			continue

print('Training complete, exiting.')									  
write_pickle(alpha_t,(C.output_boost+'_alpha_t_{0:0=2d}.file'.format(len(alpha_t))))
write_pickle(weight_dict,(C.output_boost+'_weight_dict_{0:0=2d}.file'.format(len(alpha_t))))																																	   
