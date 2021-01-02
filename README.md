For more information about the context of this project, please have a look at https://notilyze.com/news/improving-object-detection-with-contrast-stretching/ and other pages on this domain.

Note that this project has mainly been used for research objectives. Therefore the workflow is not optimized completely (i.e. running one script and having outputs directly). This has the advantage that it is clear what happens in each part of the code, and also it is easier to e.g. continue training from a certain point, add an extra (or maybe an own) preprocessing step before training etc. So more flexibility in developing new methods is created by cutting the workflow into smaller tasks.

This project has been created by combining 3 existing Github projects and adding files to connect all of them. The three original projects are:
- kbardool (keras-frcnn): https://github.com/kbardool/keras-frcnn: This part of the code trains the model, after which it can also generate detections on new images (test_frcnn)
- image_bbox_slicer: https://github.com/acl21/image_bbox_slicer: This project is used to transform images of arbitrary size to 400x400 pixel image slices. It also transforms the .xml files with ground truth labels in it accordingly.
- Object-Detection-Metrics: https://github.com/rafaelpadilla/Object-Detection-Metrics: This project is used to assess the performance of the model generated by the Faster R-CNN model. It uses both ground truth results and the detections from test_frcnn to calculate a mAP value. Also a combined file is made with all True Positives, False Positives and False Negatives (plot_boxes_white_tent_0.50.txt is an example), for research purposes and to plot these groups of detections into different colors in the original images.

Edits have been made to all three projects to minimize the need of changing formats in data. Also in that process some changes were made that improve speed performance in Object-Detection-Metrics and some changes are made in image_bbox_slicer, to make sure that objects that are sliced into more than 2 tiles, are still having correct labels in all of the resulting slices. However, note that ideally whole objects are in one slice. If objects are consequently larger than 400x400 pixels, image_bbox_slicer has a function 'resize' to resample the image size.

To avoid clashes in package versions, three environments should be created within Anaconda (use the .yml files for easy installation):
- kbardool: To do everything related to the original kbardool repository: training and testing the algorithm
- slice: To do everything related to image_bbox_slicer
- opencv: To do everything related to Object-Detection-Metrics and to run all files that connect the three original repositories (names of these .py-files start with 'opencv').

Please have a look at 'Overview Image-Detection.svg' for an overview of all python files and their relation to each other.

Please have a look at 'Thesis - Methodology.pdf' for the theory behind the code.

Some things that you might run into:
- Input images that were used in this project were all .png images. 
- When using the Object Metrics mAP code, make sure the folder with ground truth .txt files (created with opencv_04, gives one .txt file per slice) does not contain the created complete training .txt file with all labels in it (also from cv_04), as this will contaminate the mAP score (calculated with opencv_15) with a lot of false negatives.
- opencv_18: background labels ('bg') are only useful to be made in the training data set, because the validation is made in such a way that 'bg' labels are ignored; we are only interested in how good the model can detect the objects we are interested in. However, during training these 'bg' labels can cause the model to train faster as it also gets input on what he is not supposed to detect as being an object. (To let this work properly, it is important that the background labels are explicitly called 'bg'.)
- opencv_18: background labels are made by using image slices that do not contain any objects. These image slices are completely labelled as background, that is, a 400x400 pixel bounding box covering the whole empty image slice. This is different from using the mined hard negatives as is done via opencv_19.
- kbardool_14_test_frcnn_boost.py: it is recommended to run this file on batches of image slices of around 100-200 slices each time (for 10 RPN 'trees'). Due to the fact that the image results on each RPN tree needs to be saved, and then combined for the CLS part, the program slows down with growing amounts of data in memory.


For questions related to Notilyze or for requests on additional information, please contact web [at] notilyze [dot] com
