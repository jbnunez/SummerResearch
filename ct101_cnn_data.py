#make_ct101_cnn_samples.py
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
c = len(categories)
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
			#if C >= 50:
			#	break
del categories

print("==> Read in "+str(c)+" image categories")
print("==> Padding images")

#pad images
shapes1 = [image.shape[0] for image in images]
shapes2 = [image.shape[1] for image in images]
dim1 = np.max(shapes1)
dim2 = np.max(shapes2)
print("Padded dimensions:f",dim1, dim2)
#N = 5.0/0
N = len(images)
padded = np.zeros((N, dim1, dim2, 3),dtype=np.uint8)
del shapes1
del shapes2

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
del images 
print('==> Saving as .npy')

np.save("images", padded)
del padded
print('==> saved '+str(N)+' images')
