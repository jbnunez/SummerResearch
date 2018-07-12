#cancer_autoencoder_IDs.py

import os
import pickle
import pydicom

#this file generates a pickle of file names to be used in the generator
#that is used to train the cancer image autoencoder
IDs = []

cancer_path = '../desktop/cancer/TCGA-GBM/'
cpath_len = len(cancer_path)
print('==> Getting image names')
#get image from each sample


def explore_dir(path):
    ID_list = []
    contents = os.listdir(path=path)
    for obj in contents:
        if obj[-4:] == ".dcm":
            try:
                full = path+'/'+obj
                ds = pydicom.dcmread(full)
                temp = ds.pixel_array
                ID_list.append(full[cpath_len:])
            #print(len(ID_list))
            except:
                continue
        else:
            newpath = path+'/'+obj
            ID_list = ID_list + explore_dir(path=newpath)

    return ID_list



cases = os.listdir(path=cancer_path)
N_cases = len(cases)

#iterate across cases
for i in range(N_cases):
    if cases[i]=='.DS_Store':
        continue
    case_path = cancer_path+'/'+cases[i]
    subdirs = os.listdir(path=case_path)
    case_IDs = explore_dir(case_path)
    if (i+1)%25==0:
        print("-- Explored "+str(i+1)+" out of "+str(N_cases)+" subdirectories")
    IDs += case_IDs
    #print(len(IDs))

fileObject = open("cancer_IDs",'wb') 
pickle.dump(IDs, fileObject)

