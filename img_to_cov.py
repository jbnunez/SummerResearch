#img_to_cov.py
import numpy as np
import pydicom
from skimage.transform import resize

img_dim = (128, 128)

def dcm_to_array(path):
    ds = pydicom.dcmread(path)
    temp = ds.pixel_array
    resized = resize(temp, img_dim, mode='constant')
    if resized.shape == (128,128,3):
        resized = self.rgbToGray(resized)
    if resized.shape != (128, 128, 1) and resized.shape != (128, 128):
        print("unrecognized shape", resized.shape)
    resized = resized.reshape(128,128).astype(np.float32)

    return resized


def convolutions(im_mat, window=(4,4), step=(1,1)):
    #the window convolves across the image, taking the subwindow as
    #
    im_d1, im_d2 = im_mat.shape
    w_d1, w_d2 = window
    s_d1, s_d2 = step
    if (im_d1 - w_d1 + 1)%s_d1!=0 or (im_d2 - w_d2 + 1)%s_d2!=0:
        raise ValueError("Step size will result in index out of bounds")
    conv_d1 = (im_d1 - w_d1)//s_d1
    conv_d2 = (im_d2 - w_d2)//s_d2

    conv_values = np.empty((conv_d1*conv_d2, w_d1*w_d2), 
        dtype=np.float32)

    for i in range(conv_d1):
        for j in range(conv_d2):
            conv_values[i*conv_d1+j] = im_mat[i*s_d1, j*s_d2].flatten()

    return conv_values


def im_to_cov(im_mat, window=(4,4), step=(1,1)):
    convs = convolutions(im_mat, window=window, step=step)
    #cov = np.cov(convs.T)
    #print(cov.shape)
    #return cov
    return np.cov(convs.T)

#unfinished function
def sub_cov_mats(im_mat, window=(17,17), subwindow=(4,4), 
    window_step=(4,4), subwindow_step=(1,1)):
    #the window convolves across the image, taking the subwindow as the vectors
    #
    im_d1, im_d2 = im_mat.shape
    w_d1, w_d2 = window
    ws_d1, ws_d2 = window_step
    if (im_d1 - w_d1 + 1)%s_d1!=0 or (im_d2 - w_d2 + 1)%s_d2!=0:
        raise ValueError("Step size will result in index out of bounds")
    conv_d1 = (im_d1 - w_d1)//s_d1
    conv_d2 = (im_d2 - w_d2)//s_d2

    dim = ws_d1*ws_d2
    cov_mats = np.empty((conv_d1, conv_d2, dim, dim), 
        dtype=np.float32)

    for i in range(conv_d1):
        for j in range(conv_d2):
            cmat = im_to_cov(im_mat[i*s_d1, j*s_d2], 
                window=subwindow, step=subwindow_step)
            if np.linalg.matrix_rank(cmats)<dim:
                print("undiagonalizable matrix encountered")
                cov_mats[i,j,:,:] = np.eye(dim)
            else:
                cov_mats[i,j,:,:] = cmat


    return cov_mats

















