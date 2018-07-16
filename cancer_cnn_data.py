#cancer_cnn_data.py
import os
import pandas as pd
import numpy as np
import pydicom

print('==> Getting image labels')
#tumor status file
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
			#print(data.shape)
			images.append(data)
			break

		else:
			#enter deeper directory
			current_dir = os.listdir(path=path)

#pad the data
shapes1 = [image.shape[0] for image in images]
shapes2 = [image.shape[1] for image in images]
dim1 = np.max(shapes1)
dim2 = np.max(shapes2)
N = len(images)
padded = np.zeros((N, dim1, dim2))

for i in range(N):
	idim1, idim2 = images[i].shape
	diff1 = dim1-idim1
	diff2 = dim2-idim2
	start1 = diff1//2
	start2 = diff2//2
	padded[i, start1:start1+idim1, start2:start2+idim2] = images[i]



labels = np.array(labels)
targets = np.zeros((N, 2))
keep = []
for i in range(len(labels)):
	if labels[i] == "WITH TUMOR":
		targets[i,0] = 1
		keep.append(i)
	elif labels[i] == "TUMOR FREE":
		targets[i,1] = 1
		keep.append(i)
keep = np.array(keep)
padded = padded[keep]
targets = targets[keep]
N, height, width = padded.shape
images = padded.reshape((N, height, width, 1))


print('==> Loaded '+str(N)+' images')