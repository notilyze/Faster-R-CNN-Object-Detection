This project has been created by combining 3 existing Github projects and adding files to connect all of them. The three original projects are:
- kbardool (keras-frcnn): https://github.com/kbardool/keras-frcnn
- image_bbox_slicer: https://github.com/acl21/image_bbox_slicer
- Object-Detection-Metrics: https://github.com/rafaelpadilla/Object-Detection-Metrics

I have made edits to all three projects to make it all work. Also I did some changes that improved speed performances in Object-Detection-Metrics and I made some changes such that objects that are sliced into more than 2 boxes, are still having correct labels in all of the resulting slices.

To avoid clashes in package versions, I created three environments within Anaconda:
- kbardool: To do everything related to the original kbardool repository: training and testing the algorithm
- slice: To do everything related to image_bbox_slicer
- opencv: To do everything related to Object-Detection-Metrics and to run all files that connect the three original repositories.

Please have a look at Overview Image-Detection.svg for an overview of all python files and their relation to each other.

Some things that you might run into:
- Input images that were used in this project were all .png images. 
- When using the Object Metrics mAP code, make sure the folder with ground truth .txt files (created with opencv_04, gives one .txt file per slice) does not contain the created complete training .txt file with all labels in it (also from cv_04), as this will contaminate the mAP score (calculated with opencv_15) with a lot of false negatives.

opencv_18: background labels ('bg') are only useful to be made in the training data set, because the validation is made in such a way that 'bg' labels are ignored; we are only interested in how good the model can detect the objects we are interested in. However, during training these 'bg' labels can cause the model to train faster as it also gets input on what he is not supposed to detect as being an object. (To let this work properly, it is important that the background labels are explicitly called 'bg'.)

opencv_18: background labels are made by using image slices that do not contain any objects. These image slices are completely labelled as background, that is, a 400x400 pixel bounding box covering the whole empty image slice. This is different from using the mined hard negatives as is done via opencv_19
