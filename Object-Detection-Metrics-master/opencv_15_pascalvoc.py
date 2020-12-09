###########################################################################################
#																						 #
# This sample shows how to evaluate object detections applying the following metrics:	 #
#  * Precision x Recall curve	   ---->	   used by VOC PASCAL 2012)				  #
#  * Average Precision (AP)		 ---->	   used by VOC PASCAL 2012)				  #
#																						 #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)							   #
#		SMT - Signal Multimedia and Telecommunications Lab							   #
#		COPPE - Universidade Federal do Rio de Janeiro								   #
#		Last modification: Oct 9th 2018												 #
###########################################################################################

import argparse
import glob
import os
import shutil
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat

import pandas as pd
from datetime import datetime

# Validate formats
def ValidateFormats(argFormat, argName, errors):
	if argFormat == 'xywh':
		return BBFormat.XYWH
	elif argFormat == 'xyrb':
		return BBFormat.XYX2Y2
	elif argFormat is None:
		return BBFormat.XYWH  # default when nothing is passed
	else:
		errors.append(
			'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' % argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
	if arg is None:
		errors.append('argument %s: required argument' % argName)
	else:
		return True


def ValidateImageSize(arg, argName, argInformed, errors):
	errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
	ret = None
	if arg is None:
		errors.append(errorMsg)
	else:
		arg = arg.replace('(', '').replace(')', '')
		args = arg.split(',')
		if len(args) != 2:
			errors.append(
				'%s. It must be in the format \'width,height\' (e.g. \'600,400\')' % errorMsg)
		else:
			if not args[0].isdigit() or not args[1].isdigit():
				errors.append(
					'%s. It must be in INTEGER the format \'width,height\' (e.g. \'600,400\')' %
					errorMsg)
			else:
				ret = (int(args[0]), int(args[1]))
	return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
	if arg == 'abs':
		return CoordinatesType.Absolute
	elif arg == 'rel':
		return CoordinatesType.Relative
	elif arg is None:
		return CoordinatesType.Absolute  # default when nothing is passed
	errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
	if arg is None:
		errors.append('argument %s: invalid directory' % nameArg)
	elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
		errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
	# elif os.path.isdir(os.path.join(currentPath, arg)) is True:
	#	 arg = os.path.join(currentPath, arg)
	else:
		arg = os.path.join(currentPath, arg)
	return arg


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


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1 (beta)'

parser = argparse.ArgumentParser(
	prog='Object Detection Metrics - Pascal VOC',
	description='This project applies the most popular metrics used to evaluate object detection '
	'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
	'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
	epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# formatter_class=RawTextHelpFormatter)
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument(
	'-gt',
	'--gtfolder',
	dest='gtFolder',
	default=os.path.join(currentPath, 'groundtruths'),
	metavar='',
	help='folder containing your ground truth bounding boxes')
parser.add_argument(
	'-det',
	'--detfolder',
	dest='detFolder',
	default=os.path.join(currentPath, 'detections'),
	metavar='',
	help='folder containing your detected bounding boxes')
parser.add_argument(
	'-sp', 
	'--savePath', 
	dest='savePath', 
	default=os.path.join(currentPath, 'results'),
	metavar='', 
	help='folder where the plots are saved')

# Optional
parser.add_argument(
	'-t',
	'--threshold',
	dest='iouThreshold',
	type=float,
	default=0.5,
	metavar='',
	help='IOU threshold. Default 0.5')
parser.add_argument(
	'-gtformat',
	dest='gtFormat',
	metavar='',
	default='xywh',
	help='format of the coordinates of the ground truth bounding boxes: '
	'(\'xywh\': <left> <top> <width> <height>)'
	' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
	'-detformat',
	dest='detFormat',
	metavar='',
	default='xywh',
	help='format of the coordinates of the detected bounding boxes '
	'(\'xywh\': <left> <top> <width> <height>) '
	'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument(
	'-gtcoords',
	dest='gtCoordinates',
	default='abs',
	metavar='',
	help='reference of the ground truth bounding box coordinates: absolute '
	'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
	'-detcoords',
	default='abs',
	dest='detCoordinates',
	metavar='',
	help='reference of the ground truth bounding box coordinates: '
	'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument(
	'-imgsize',
	dest='imgSize',
	metavar='',
	help='image size. Required if -gtcoords or -detcoords are \'rel\'')

parser.add_argument(
	'-np',
	'--noplot',
	dest='showPlot',
	action='store_false',
	help='no plot is shown during execution')

args = parser.parse_args()


iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
	gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
else:
	# errors.pop()
	gtFolder = os.path.join(currentPath, 'groundtruths')
	if os.path.isdir(gtFolder) is False:
		errors.append('folder %s not found' % gtFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
	imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
	imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)


detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
savePath = ValidatePaths(args.savePath, '--savePath', errors)


# Create directory to save results
os.makedirs(savePath,exist_ok=True)
# Show plot during execution
showPlot = args.showPlot

print('gtFolder = %s' % gtFolder)

startTime = datetime.now()
# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
	gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(
	detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
allClasses.sort()

evaluator = Evaluator()
acc_AP = 0
validClasses = 0

# Plot Precision x Recall curve
detections = evaluator.PlotPrecisionRecallCurve(
	allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
	IOUThreshold=iouThreshold,  # IOU threshold
	method=MethodAveragePrecision.EveryPointInterpolation,
	showAP=True,  # Show Average Precision in the title of the plot
	showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
	savePath=savePath,
	showGraphic=showPlot)

f = open(os.path.join(savePath, 'results_IoU_{}.txt'.format(iouThreshold)), 'w')
f.write('Object Detection Metrics\n')
f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
f.write('Average Precision (AP), Precision and Recall per class:')

# each detection is a class
for metricsPerClass in detections:

	# Get metric values per each class
	cl = metricsPerClass['class']
	ap = metricsPerClass['AP']
	precision = metricsPerClass['precision']
	recall = metricsPerClass['recall']
	totalPositives = metricsPerClass['total positives']
	total_TP = metricsPerClass['total TP']
	total_FP = metricsPerClass['total FP']
	area_sq = metricsPerClass['area_sq']
	longest_side = metricsPerClass['longest_side']
	gt_counts = metricsPerClass['Count_gt_detections']
	dictionary = metricsPerClass['GT_dict']
	f.write("Class: {}, Total Positives: {}\n".format(cl,totalPositives))
	

	if totalPositives > 0:
		validClasses = validClasses + 1
		acc_AP = acc_AP + ap
		prec = ['%.2f' % p for p in precision]
		rec = ['%.2f' % r for r in recall]
		ap_str = "{0:.2f}%".format(ap * 100)
		# ap_str = "{0:.4f}%".format(ap * 100)
		print('AP: %s (%s)' % (ap_str, cl))
		f.write('\n\nClass: %s' % cl)
		f.write('\nAP: %s' % ap_str)
		f.write('\nPrecision: %s' % prec)
		f.write('\nRecall: %s' % rec)
		a_sq = ['%.2f' % a for a in area_sq]
		f.write('\nArea_sq: %s' % a_sq)
		l_side = ['%.1f' % l for l in longest_side]
		f.write('\n\nLongest_side: %s' % l_side)
		f.write('\nNumber of ground-truth boxes: {} should be equal to len(gt_counts)= {}\n'.format(totalPositives,len(gt_counts)))
		num_gt_not_found=sum(int(num) < 1 for num in gt_counts)+(totalPositives-len(gt_counts))
		f.write('Amount of gt boxes not found = {}\n'.format(num_gt_not_found))
		num_gt_found=sum(int(num) > 0 for num in gt_counts)
		f.write('Amount of gt boxes found = {}\n'.format(num_gt_found))
		double_boxes=sum(gt_counts)-num_gt_found
		f.write('Amount of double boxes = {}\n'.format(double_boxes))
		f.write('Amount of detections that did not have (enough) overlap with any gt box = {}\n'.format(total_FP-double_boxes))
		f.write('\nGT_counts: %s' % gt_counts)
		gt_detect=pd.DataFrame.from_dict(dictionary,orient='index').transpose()
		gt_detect.to_csv(os.path.join(savePath, 'detected_gt_analysis_{}_{}.csv'.format(iouThreshold,cl)),index=True,index_label='Index')
		
		

mAP = acc_AP / validClasses
mAP_str = "{0:.2f}%".format(mAP * 100)
print('mAP: %s' % mAP_str)
f.write('\n\n\nmAP: %s' % mAP_str)
f.close()
print(type(prec))
print(type(area_sq.tolist()))
print("gtFolder: {}\n".format(gtFolder))
print("detFolder: {}\n".format(detFolder))
print("savePath: {}\n".format(savePath))
print(datetime.now() - startTime)