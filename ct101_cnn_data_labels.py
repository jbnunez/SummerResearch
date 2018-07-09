#make_ct101_cnn_labels.py
import os
import numpy as np
from scipy import misc

ignore_misc = True
if ignore_misc:
	print('==> Ignoring miscellaneous category')
else:
	print('==> Including miscellaneous category')

path = '../desktop/ct101/101_ObjectCategories/'
categories = os.listdir(path=path)

#read in images
images = []
labels = []
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
					labels.append(C)
			C += 1
			# if C >= 50:
			# 	break
labs = np.array(labels)
del labels
del categories

print("==> Read in "+str(C)+" image categories")
print("==> Making one-hot targets")

eye = np.eye(C)
targets = eye[labs]
del labs
eye=None
print("==> Saving one-hot targets")



np.save("targets", targets)

print('==> Saved '+str(len(targets))+' labels')
