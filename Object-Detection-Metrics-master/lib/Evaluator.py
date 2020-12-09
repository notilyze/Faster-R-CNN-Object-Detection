###########################################################################################
#																						 #
# Evaluator class: Implements the most popular metrics for object detection			   #
#																						 #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)							   #
#		SMT - Signal Multimedia and Telecommunications Lab							   #
#		COPPE - Universidade Federal do Rio de Janeiro								   #
#		Last modification: Oct 9th 2018												 #
###########################################################################################

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from BoundingBox import *
from BoundingBoxes import *
from utils import *

import pandas as pd
from operator import itemgetter
from itertools import groupby
from operator import add

import bisect

class Evaluator:
	def GetPascalVOCMetrics(self,
							boundingboxes,
							IOUThreshold=0.5,
							method=MethodAveragePrecision.EveryPointInterpolation,
							savePath=None):
		"""Get the metrics used by the VOC Pascal 2012 challenge.
		Get
		Args:
			boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
			bounding boxes;
			IOUThreshold: IOU threshold indicating which detections will be considered TP or FP
			(default value = 0.5);
			method (default = EveryPointInterpolation): It can be calculated as the implementation
			in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
			interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
			or EveryPointInterpolation"  (ElevenPointInterpolation);
		Returns:
			A list of dictionaries. Each dictionary contains information and metrics of each class.
			The keys of each dictionary are:
			dict['class']: class representing the current dictionary;
			dict['precision']: array with the precision values;
			dict['recall']: array with the recall values;
			dict['AP']: average precision;
			dict['interpolated precision']: interpolated precision values;
			dict['interpolated recall']: interpolated recall values;
			dict['total positives']: total number of ground truth positives;
			dict['total TP']: total number of True Positive detections;
			dict['total FP']: total number of False Negative detections;
		"""
		ret = []  # list containing metrics (precision, recall, average precision) of each class
		# List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
		groundTruths = []
		# List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
		detections = []
		# Get all classes
		classes = []
		# Loop through all bounding boxes and separate them into GTs and detections
		for bb in boundingboxes.getBoundingBoxes():
			# [imageName, class, confidence, (bb coordinates XYX2Y2)]
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
		# Precision x Recall is obtained individually by each class
		# Loop through by classes
		for c in classes:
			# Get only detection of class c
			dects = []
			[dects.append(d) for d in detections if d[1] == c]
			# Get only ground truths of class c
			gts = []
			[gts.append(g) for g in groundTruths if g[1] == c]
			npos = len(gts)
			gt_dict=defaultdict(list)
			for line in gts:
				details=line[0:]
				gt_dict[line[0]].append(details)
				
			# sort detections by decreasing confidence
			dects = sorted(dects, key=lambda conf: conf[2], reverse=False)
			dects.reverse()
			print("Length dects: {}".format(len(dects)))
			print("Length gts: {}".format(len(gts)))
			x_1 = np.zeros(len(dects))
			y_1 = np.zeros(len(dects))
			x_2 = np.zeros(len(dects))
			y_2 = np.zeros(len(dects))
			confidence = np.zeros(len(dects))
			TP = np.zeros(len(dects))
			FP = np.zeros(len(dects))
			Already_detected = np.zeros(len(dects))
			Count_gt_detections_all_images=[]
			# create dictionary with amount of gts for each image
			det = Counter([cc[0] for cc in gts])
			for key, val in det.items():
				det[key] = np.zeros(val)
			# print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
			# Loop through detections
			fp_train=open(os.path.join(savePath, 'hard_negative_mining_{}_{:.2f}.txt'.format(c,IOUThreshold)), 'w')
			plot_box=open(os.path.join(savePath, 'plot_boxes_{}_{:.2f}.txt'.format(c,IOUThreshold)), 'w')
			for d in range(len(dects)):
				confidence[d]=dects[d][2]
				x_1[d]=dects[d][3][0]
				y_1[d]=dects[d][3][1]
				x_2[d]=dects[d][3][2]
				y_2[d]=dects[d][3][3]
				if  dects[d][0] in 'AD_S030_06_000141':
					print(dects[d])
				# print('dect %s => %s' % (dects[d][0], dects[d][3],))
				# Find ground truth image
				gt = gt_dict[dects[d][0]]
				iouMax = sys.float_info.min
				Count_gt_detections = np.zeros(len(gt))
				for j in range(len(gt)):
					# print('Ground truth gt => %s' % (gt[j][3],))
					iou = Evaluator.iou(dects[d][3], gt[j][3])
					if iou > iouMax:
						iouMax = iou
						jmax = j
				# Assign detection as true positive/don't care/false positive
				if iouMax >= IOUThreshold:
					Count_gt_detections[jmax] += 1
					if det[dects[d][0]][jmax] == 0:
						if dects[d][0] in 'AD_S030_06_000141':
							print("TP found")
						TP[d] = 1  # count as true positive
						det[dects[d][0]][jmax] = 1  # flag as already 'seen'
						if dects[d][0] in 'AD_S030_06_000141':
							print(det[dects[d][0]])
						plot_box.write(dects[d][0]+","+dects[d][1]+","+str(dects[d][2])+","+str(int(x_1[d]))+","+str(int(y_1[d]))+","+str(int(x_2[d]))+","+str(int(y_2[d]))+",TP\n")
					else:
						#this gt was already detected
						Already_detected[d] = 1
						FP[d] = 1  # count as false positive
						plot_box.write(dects[d][0]+","+dects[d][1]+","+str(dects[d][2])+","+str(int(x_1[d]))+","+str(int(y_1[d]))+","+str(int(x_2[d]))+","+str(int(y_2[d]))+",Double\n")
				# - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
				else:
					FP[d] = 1  # count as false positive
					fp_train.write("folder/"+dects[d][0]+".png,"+str(int(x_1[d]))+","+str(int(y_1[d]))+","+str(int(x_2[d]))+","+str(int(y_2[d]))+",bg\n")
					plot_box.write(dects[d][0]+","+dects[d][1]+","+str(dects[d][2])+","+str(int(x_1[d]))+","+str(int(y_1[d]))+","+str(int(x_2[d]))+","+str(int(y_2[d]))+",FP\n")
				if len(gt)>0:
					Count_gt_detections_all_images.append([gt[0][0]]+Count_gt_detections.tolist())
			# compute precision, recall and average precision
			fp_train.close()
			key=itemgetter(0)
			Count_gt_detections_all_images.sort(key=key)
			data={}
			
			for l2 in Count_gt_detections_all_images:
				if l2[0] in data:
					list_new=list(map(add,l2[1:],data[l2[0]]))
					data[l2[0]]=list_new
				else:
					data[l2[0]] = l2[1:]
			
			
			count_gt_pic=0
			SSID_slice=gts[0][0]
			for g in range(len(gts)):
				if gts[g][0] not in SSID_slice:
					count_gt_pic=0
					SSID_slice=gts[g][0]
				
				gt_box=det[gts[g][0]][count_gt_pic]
				if isinstance(gt_box, int) or isinstance(gt_box, float):
					if gt_box<0.5:
						plot_box.write(gts[g][0]+","+gts[g][1]+","+str(gts[g][2])+","+str(int(gts[g][3][0]))+","+str(int(gts[g][3][1]))+","+str(int(gts[g][3][2]))+","+str(int(gts[g][3][3]))+",FN\n")
				else:
					if max(gt_box)<0.5:
						plot_box.write(gts[g][0]+","+gts[g][1]+","+str(gts[g][2])+","+str(int(gts[g][3][0]))+","+str(int(gts[g][3][1]))+","+str(int(gts[g][3][2]))+","+str(int(gts[g][3][3]))+",FN\n")
				count_gt_pic+=1
			list_values = [ v for v in data.values() ]
			flat_list = [item for sublist in list_values for item in sublist]
			int_list = [ int(x) for x in flat_list ]

			acc_FP = np.cumsum(FP)
			acc_TP = np.cumsum(TP)
			rec = acc_TP / npos
			prec = np.divide(acc_TP, (acc_FP + acc_TP))
			area_sq = np.sqrt(np.multiply(x_2-x_1,y_2-y_1))
			longest_side = np.maximum(x_2-x_1,y_2-y_1)
			# Depending on the method, call the right implementation
			if method == MethodAveragePrecision.EveryPointInterpolation:
				[ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
			else:
				[ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
			# add class result in the dictionary to be returned
			r = {
				'class': c,
				'precision': prec,
				'recall': rec,
				'AP': ap,
				'interpolated precision': mpre,
				'interpolated recall': mrec,
				'total positives': npos,
				'total TP': np.sum(TP),
				'total FP': np.sum(FP),
				'area_sq': area_sq,
				'longest_side': longest_side,
				'Already_detected': Already_detected,
				'Count_gt_detections': int_list,
				'GT_dict' : data
			}
			ret.append(r)
		return ret

	def PlotPrecisionRecallCurve(self,
								 boundingBoxes,
								 IOUThreshold=0.5,
								 method=MethodAveragePrecision.EveryPointInterpolation,
								 showAP=False,
								 showInterpolatedPrecision=False,
								 savePath=None,
								 showGraphic=True):
		"""PlotPrecisionRecallCurve
		Plot the Precision x Recall curve for a given class.
		Args:
			boundingBoxes: Object of the class BoundingBoxes representing ground truth and detected
			bounding boxes;
			IOUThreshold (optional): IOU threshold indicating which detections will be considered
			TP or FP (default value = 0.5);
			method (default = EveryPointInterpolation): It can be calculated as the implementation
			in the official PASCAL VOC toolkit (EveryPointInterpolation), or applying the 11-point
			interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
			or EveryPointInterpolation"  (ElevenPointInterpolation).
			showAP (optional): if True, the average precision value will be shown in the title of
			the graph (default = False);
			showInterpolatedPrecision (optional): if True, it will show in the plot the interpolated
			 precision (default = False);
			savePath (optional): if informed, the plot will be saved as an image in this path
			(ex: /home/mywork/ap.png) (default = None);
			showGraphic (optional): if True, the plot will be shown (default = True)
		Returns:
			A list of dictionaries. Each dictionary contains information and metrics of each class.
			The keys of each dictionary are:
			dict['class']: class representing the current dictionary;
			dict['precision']: array with the precision values;
			dict['recall']: array with the recall values;
			dict['AP']: average precision;
			dict['interpolated precision']: interpolated precision values;
			dict['interpolated recall']: interpolated recall values;
			dict['total positives']: total number of ground truth positives;
			dict['total TP']: total number of True Positive detections;
			dict['total FP']: total number of False Positive detections;
		"""
		results = self.GetPascalVOCMetrics(boundingBoxes, IOUThreshold, method, savePath)
		result = None
		# Each result represents a class
		for result in results:
			if result is None:
				raise IOError('Error: Class %d could not be found.' % classId)

			classId = result['class']
			precision = result['precision']
			recall = result['recall']
			average_precision = result['AP']
			mpre = result['interpolated precision']
			mrec = result['interpolated recall']
			npos = result['total positives']
			total_tp = result['total TP']
			total_fp = result['total FP']
			area_sq = result['area_sq']
			longest_side = result['longest_side']
			
			plt.close()
			if showInterpolatedPrecision:
				if method == MethodAveragePrecision.EveryPointInterpolation:
					plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
				elif method == MethodAveragePrecision.ElevenPointInterpolation:
					# Uncomment the line below if you want to plot the area
					# plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
					# Remove duplicates, getting only the highest precision of each recall value
					nrec = []
					nprec = []
					for idx in range(len(mrec)):
						r = mrec[idx]
						if r not in nrec:
							idxEq = np.argwhere(mrec == r)
							nrec.append(r)
							nprec.append(max([mpre[int(id)] for id in idxEq]))
					plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
			fig,ax1=plt.subplots()
			ax1.set_xlabel('recall')
			ax1.set_ylabel('precision')
			ln1=ax1.plot(recall, precision, label='Precision')
			#Shrink current plot by 20% so legend can be put next to it
			box = ax1.get_position()
			ax1.set_position([box.x0, box.y0 + box.height * 0.1,
				 box.width, box.height * 0.9])
			
			ax2=ax1.twinx()
			ax2.set_ylabel('sqrt(Area)')
			ln2=ax2.plot(recall,pd.Series(longest_side).rolling(window=100).mean(),color='red',label='Rolling mean Sqrt(Area)')

			box = ax2.get_position()
			ax2.set_position([box.x0, box.y0 + box.height * 0.15,
				 box.width, box.height * 0.85])
			
			lns=ln1+ln2
			labels_all=[l.get_label() for l in lns]
			ax1.legend(lns, labels_all, loc='upper center', bbox_to_anchor=(0.5, -0.05),
		  fancybox=True, shadow=True)

			plt.grid()
			if showAP:
				ap_str = "{0:.2f}%".format(average_precision * 100)
				plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
			else:
				plt.title('Precision x Recall curve \nClass: %s' % str(classId))
			
				
				
			############################################################
			# Uncomment the following block to create plot with points #
			############################################################
			# plt.plot(recall, precision, 'bo')
			# labels = ['R', 'Y', 'J', 'A', 'U', 'C', 'M', 'F', 'D', 'B', 'H', 'P', 'E', 'X', 'N', 'T',
			# 'K', 'Q', 'V', 'I', 'L', 'S', 'G', 'O']
			# dicPosition = {}
			# dicPosition['left_zero'] = (-30,0)
			# dicPosition['left_zero_slight'] = (-30,-10)
			# dicPosition['right_zero'] = (30,0)
			# dicPosition['left_up'] = (-30,20)
			# dicPosition['left_down'] = (-30,-25)
			# dicPosition['right_up'] = (20,20)
			# dicPosition['right_down'] = (20,-20)
			# dicPosition['up_zero'] = (0,30)
			# dicPosition['up_right'] = (0,30)
			# dicPosition['left_zero_long'] = (-60,-2)
			# dicPosition['down_zero'] = (-2,-30)
			# vecPositions = [
			#	 dicPosition['left_down'],
			#	 dicPosition['left_zero'],
			#	 dicPosition['right_zero'],
			#	 dicPosition['right_zero'],  #'R', 'Y', 'J', 'A',
			#	 dicPosition['left_up'],
			#	 dicPosition['left_up'],
			#	 dicPosition['right_up'],
			#	 dicPosition['left_up'],  # 'U', 'C', 'M', 'F',
			#	 dicPosition['left_zero'],
			#	 dicPosition['right_up'],
			#	 dicPosition['right_down'],
			#	 dicPosition['down_zero'],  #'D', 'B', 'H', 'P'
			#	 dicPosition['left_up'],
			#	 dicPosition['up_zero'],
			#	 dicPosition['right_up'],
			#	 dicPosition['left_up'],  # 'E', 'X', 'N', 'T',
			#	 dicPosition['left_zero'],
			#	 dicPosition['right_zero'],
			#	 dicPosition['left_zero_long'],
			#	 dicPosition['left_zero_slight'],  # 'K', 'Q', 'V', 'I',
			#	 dicPosition['right_down'],
			#	 dicPosition['left_down'],
			#	 dicPosition['right_up'],
			#	 dicPosition['down_zero']
			# ]  # 'L', 'S', 'G', 'O'
			# for idx in range(len(labels)):
			#	 box = dict(boxstyle='round,pad=.5',facecolor='yellow',alpha=0.5)
			#	 plt.annotate(labels[idx],
			#				 xy=(recall[idx],precision[idx]), xycoords='data',
			#				 xytext=vecPositions[idx], textcoords='offset points',
			#				 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
			#				 bbox=box)
			if savePath is not None:
				fig.savefig(os.path.join(savePath, classId + '_IoU_{}.png'.format(IOUThreshold)))
			if showGraphic is True:
				fig.show()
				# plt.waitforbuttonpress()
				plt.pause(0.05)
		return results

	@staticmethod
	def CalculateAveragePrecision(rec, prec):
		mrec = []
		mrec.append(0)
		[mrec.append(e) for e in rec]
		mrec.append(1)
		mpre = []
		mpre.append(0)
		[mpre.append(e) for e in prec]
		mpre.append(0)
		for i in range(len(mpre) - 1, 0, -1):
			mpre[i - 1] = max(mpre[i - 1], mpre[i])
		ii = []
		for i in range(len(mrec) - 1):
			if mrec[1:][i] != mrec[0:-1][i]:
				ii.append(i + 1)
		ap = 0
		for i in ii:
			ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
		# return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
		return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

	@staticmethod
	# 11-point interpolated average precision
	def ElevenPointInterpolatedAP(rec, prec):
		# def CalculateAveragePrecision2(rec, prec):
		mrec = []
		# mrec.append(0)
		[mrec.append(e) for e in rec]
		# mrec.append(1)
		mpre = []
		# mpre.append(0)
		[mpre.append(e) for e in prec]
		# mpre.append(0)
		recallValues = np.linspace(0, 1, 11)
		recallValues = list(recallValues[::-1])
		rhoInterp = []
		recallValid = []
		# For each recallValues (0, 0.1, 0.2, ... , 1)
		for r in recallValues:
			# Obtain all recall values higher or equal than r
			argGreaterRecalls = np.argwhere(mrec[:] >= r)
			pmax = 0
			# If there are recalls above r
			if argGreaterRecalls.size != 0:
				pmax = max(mpre[argGreaterRecalls.min():])
			recallValid.append(r)
			rhoInterp.append(pmax)
		# By definition AP = sum(max(precision whose recall is above r))/11
		ap = sum(rhoInterp) / 11
		# Generating values for the plot
		rvals = []
		rvals.append(recallValid[0])
		[rvals.append(e) for e in recallValid]
		rvals.append(0)
		pvals = []
		pvals.append(0)
		[pvals.append(e) for e in rhoInterp]
		pvals.append(0)
		# rhoInterp = rhoInterp[::-1]
		cc = []
		for i in range(len(rvals)):
			p = (rvals[i], pvals[i - 1])
			if p not in cc:
				cc.append(p)
			p = (rvals[i], pvals[i])
			if p not in cc:
				cc.append(p)
		recallValues = [i[0] for i in cc]
		rhoInterp = [i[1] for i in cc]
		return [ap, rhoInterp, recallValues, None]

	# For each detections, calculate IOU with reference
	@staticmethod
	def _getAllIOUs(reference, detections):
		ret = []
		bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
		# img = np.zeros((200,200,3), np.uint8)
		for d in detections:
			bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
			iou = Evaluator.iou(bbReference, bb)
			# Show blank image with the bounding boxes
			# img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
			# img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
			ret.append((iou, reference, d))  # iou, reference, detection
		# cv2.imshow("comparing",img)
		# cv2.waitKey(0)
		# cv2.destroyWindow("comparing")
		return sorted(ret, key=lambda i: i[0], reverse=True)  # sort by iou (from highest to lowest)

	@staticmethod
	def iou(boxA, boxB):
		# if boxes dont intersect
		if Evaluator._boxesIntersect(boxA, boxB) is False:
			return 0
		interArea = Evaluator._getIntersectionArea(boxA, boxB)
		union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
		# intersection over union
		iou = interArea / union
		assert iou >= 0
		return iou

	# boxA = (Ax1,Ay1,Ax2,Ay2)
	# boxB = (Bx1,By1,Bx2,By2)
	@staticmethod
	def _boxesIntersect(boxA, boxB):
		if boxA[0] > boxB[2]:
			return False  # boxA is right of boxB
		if boxB[0] > boxA[2]:
			return False  # boxA is left of boxB
		if boxA[3] < boxB[1]:
			return False  # boxA is above boxB
		if boxA[1] > boxB[3]:
			return False  # boxA is below boxB
		return True

	@staticmethod
	def _getIntersectionArea(boxA, boxB):
		xA = max(boxA[0], boxB[0])
		yA = max(boxA[1], boxB[1])
		xB = min(boxA[2], boxB[2])
		yB = min(boxA[3], boxB[3])
		# intersection area
		return (xB - xA + 1) * (yB - yA + 1)

	@staticmethod
	def _getUnionAreas(boxA, boxB, interArea=None):
		area_A = Evaluator._getArea(boxA)
		area_B = Evaluator._getArea(boxB)
		if interArea is None:
			interArea = Evaluator._getIntersectionArea(boxA, boxB)
		return float(area_A + area_B - interArea)

	@staticmethod
	def _getArea(box):
		return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
