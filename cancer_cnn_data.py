#cancer_cnn_data.py
import os
import pandas as pd
import pydicom


labels_file = "nationwidechildrens.org_clinical_follow_up_v1.0_gbm.txt"
labels_df = pd.read_csv(labels_file, sep='	', index_col=1)
labels_df = labels_df.drop(labels_df.index[[0,1]])
labels = labels_df['tumor_status']
#print("labels", labels)
#print(type(labels_df.index[0]))

indices = labels_df.index
images = []


cancer_path = '../desktop/cancer/cancer_images/'

#get image from each sample
for ind in indices:
	path = cancer_path + ind
	this = os.listdir(path=path)
	while this[0][:-4] is not '.dcm':
		this = os.listdir(path=this[0])
	#read first image
	ds = pydicom.dcmread(this[0])
	data = ds.pixel_array
	images.add(data)

images = np.array(images)
