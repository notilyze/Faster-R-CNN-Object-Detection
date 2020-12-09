# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 12:55:28 2019

@author: paul
"""

import pandas as pd
import os

# Location of csv file with some kind of IDs, so that we can split our observations into training, validation and test set. Note that this does not have to be the file with labels already, in our case we used just a table with one column with unique IDs, which we could later on use to select the right labels for the right task (training, validation, testing).
src=r'opencv_input\image_ids.csv'

# Name of ID column
id_column='SSID'

# Location to write split data
dst=r'opencv_output'
os.makedirs(dst,exist_ok=True)

# Read data
all_data=pd.read_csv(src)


# Select training data, assign random state if you want the selection to be reproducible. 60% training, 20% validation, 20% test
df_train=all_data.sample(frac=0.6,random_state=1)
df_merge=all_data.merge(df_train.drop_duplicates(),on=[id_column],how='left',indicator=True)
df_test_val=df_merge[df_merge['_merge'] == 'left_only']
df_val=df_test_val.sample(frac=0.5,random_state=3)
df_merge2=df_test_val.merge(df_val.drop_duplicates(),on=[id_column],how='left',indicator='_merge2')
df_test=df_merge2[df_merge2['_merge2'] == 'left_only']

# Write split data to new csv files
df_train.to_csv(os.path.join(dst,'data_training.csv'),columns=[id_column],index=False)
df_val.to_csv(os.path.join(dst,'data_val.csv'),columns=[id_column],index=False)
df_test.to_csv(os.path.join(dst,'data_test.csv'),columns=[id_column],index=False)