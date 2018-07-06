#ct101_cnn_data.py
import os
import numpy as np
from scipy import misc
from sklearn import model_selection

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
					im_arr = misc.imread(dirpath+'/'+im_file)
					images.append(im_arr)
					labels.append(C)
			C += 1
labels = np.array(labels)
print("==> Read in "+str(C)+" image categories")
print("==> Padding images")


#pad images
shapes1 = [image.shape[0] for image in images]
shapes2 = [image.shape[1] for image in images]
dim1 = np.max(shapes1)
dim2 = np.max(shapes2)
N = len(images)
padded = np.zeros((N, dim1, dim2, 3))

for i in range(N):
	idim1, idim2 = images[i].shape[0:2]
	diff1 = dim1-idim1
	diff2 = dim2-idim2
	start1 = diff1//2
	start2 = diff2//2
	if len(images[i].shape) == 3:
		padded[i, start1:start1+idim1, start2:start2+idim2] = images[i]
	else:
		padded[i, start1:start1+idim1, start2:start2+idim2, 0] = images[i]
	if i%1000 == 0:
		print("==> Padded "+str(i)+" images")
images = None
print('==> Making one-hot target vectors')
#make targets into one-hot vectors
# targets = np.zeros((N, C))
# if len(labels)==N:
# 	print("yay the lengths match and are "+str(N))
# for i in range(N):
# 	targets[i][labels[i]] = 1.
# 	if i%1000 == 0:
# 		print("==> Made "+str(i)+" target vectors")
#this is the coolest bit of python golf I've seen in a while
targets = np.eye(C)[labels]

#images = np.array(padded)
print("==> Making test-train split")
X_train, X_test, y_train, y_test = model_selection.train_test_split(padded,
	targets, test_size=0.2)
targets = None
padded = None

print('==> Loaded '+str(N)+' images')
