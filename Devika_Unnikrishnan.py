## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, count_fingers.py
## 3. In this challenge, you are only permitted to import numpy, and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
import sys
#It's ok to import whatever you want from the local util module if you would like:


def breakIntoGrids(im, s = 9):
    '''
    Break overall image into overlapping grids of size s x s, s must be odd.
    '''
    grids = []

    h = s//2 #half grid size minus one.
    for i in range(h, im.shape[0]-h):
        for j in range(h, im.shape[1]-h):
            grids.append(im[i-h:i+h+1,j-h:j+h+1].ravel())

    return np.vstack(grids)


def reshapeIntoImage(vector, im_shape, s = 9):
    '''
    Reshape vector back into image. 
    '''
    h = s//2 #half grid size minus one. 
    image = np.zeros(im_shape)
    image[h:-h, h:-h] = vector.reshape(im_shape[0]-2*h, im_shape[1]-2*h)

    return image


def count_fingers(im):
    im = im > 92 #Threshold image
    X = breakIntoGrids(im, s = 9) #Break into 9x9 grids

    #Use rule we learned with decision tree
    treeRule1 = lambda X: np.logical_and(np.logical_and(X[:, 40] == 1, X[:,0] == 0), X[:, 53] == 0)
    yhat = treeRule1(X)

    #Reshape prediction ino image:
    yhat_reshaped = reshapeIntoImage(yhat, im.shape)

    labels = [1, 2, 3]
    index = np.where(yhat_reshaped == 1)
    
    m = len(yhat_reshaped)/ 2
    
    count = np.sum(yhat_reshaped[0:int(m),:])
    print(count)
    if count >= 0 and count < 109:
        return labels[0]
    elif count >= 109 and count < 150:
        return labels[1]
    else:
        return labels[2]