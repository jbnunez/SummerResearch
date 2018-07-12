#make_ct101_cnn_dict.py
import os
import numpy as np
#from scipy import misc

#this file is incomplete and was never used because a generator was not
#needed for this problem

ignore_misc = True
if ignore_misc:
	print('==> Ignoring miscellaneous category')
else:
	print('==> Including miscellaneous category')

path = '../desktop/ct101/101_ObjectCategories/'
categories = os.listdir(path=path)

#read in images
list_ids_test = []
labels_test = []
list_ids_train = []
labels_train = []
c = len(categories)
#counter for actual number of categories
C = 0
print("==> Reading in images")
for i in range(c):
	if categories[i] != '.DS_Store':
		if ignore_misc == False or categories[i] != 'BACKGROUND_Google':
			dirpath = path+'/'+categories[i]
			im_files = os.listdir(path=dirpath)
			
			for im_file in im_files:
				if im_file != '.DS_Store':
					if np.random.uniform()<0.8:
						list_ids_test.append('/'+categories[i]+'/'+im_file)
						labels_test.append(C)
					else:
						list_ids_train.append('/'+categories[i]+'/'+im_file)
						labels_train.append(C)
			C += 1
			# if C >= 50:
			# 	break
labs = np.array(labels_train)
labs2 = np.array(labels_test)

print("==> Read in "+str(C)+" image categories")
print("==> Making one-hot targets")

eye = np.eye(C)
labels = eye[labs]
del labs
del eye
print("==> Saving one-hot targets")
np.save("labels_test", labels_test)
np.save("labels_train", labels_train)


print('==> Saved '+str(len(labels))+' labels')
