#cancer_cnn_data.py
import os
import pandas as pd
import numpy as np
import pydicom

print('==> Getting image labels')
labels_file = "nationwidechildrens.org_clinical_follow_up_v1.0_gbm.txt"
labels_df = pd.read_csv(labels_file, sep='	', index_col=1)
labels_df = labels_df.drop(labels_df.index[[0,1]])
labels = labels_df['tumor_status']
#print("labels", labels)
#print(type(labels_df.index[0]))

indices = labels_df.index
images = []


cancer_path = '../desktop/cancer/TCGA-GBM/'
print('==> Getting images')
#get image from each sample
for ind in indices:
	path = cancer_path + ind
	current_dir = os.listdir(path=path)
	#get into deeper directories
	while current_dir[0][:-4]!='.dcm':
		#skip non-directory, nonimage files
		i = 0
		while current_dir[i]=='.DS_Store':
			i += 1
		#add new directory to path
		path += '/' + current_dir[i]
		if current_dir[i]=='000000.dcm':
			#read first image
			ds = pydicom.dcmread(path)
			data = ds.pixel_array
			images.append(data)
			break

		else:
			#enter deeper directory
			current_dir = os.listdir(path=path)

images = np.array(images)
labels = np.array(labels)
print('==> Loaded '+str(images.shape[0])+' images')