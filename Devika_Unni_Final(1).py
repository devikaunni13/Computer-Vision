#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from tqdm import tqdm
import time
import os


# In[2]:





# In[18]:


def train(path_to_images, csv_file):
    '''
    First method you need to complete. 
    Args: 
    path_to_images = path to jpg image files
    csv_file = path and filename to csv file containing frame numbers and steering angles. 
    Returns: 
    NN = Trained Neural Network object 
    '''
    
     # Import Steering Angles CSV
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]
    NN = NeuralNetwork()
    
        
    global min_st_angle, max_st_angle
    
    epochs = 4100
    min_st_angle, max_st_angle = 0, 0
    max_st_angle = steering_angles.max()
    min_st_angle = steering_angles.min()
    bins = np.linspace(min_st_angle, max_st_angle+40, 32)
    steering_angles_binned = np.digitize(steering_angles, bins)
    steering_angles_matrix = np.zeros((len(steering_angles_binned),32))
    
    for i, val in enumerate(steering_angles_binned):
        steering_angles_matrix[i, val-1] = 1
    
    train_imgs = []
    lr = 0.001


    for i,frame in enumerate(frame_nums):
        img = cv2.imread(path_to_images + '/' + str(int(frame)).zfill(4) + '.jpg')
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized_img = cv2.resize(gray_image, (32,32))
        crop_img = resized_img[int(resized_img.shape[0]/2):, :]
        crop_img = crop_img/255.
        
        train_imgs.append(crop_img)
    
    train_imgs = np.reshape(train_imgs,(1500,32*16))
    
    for iter in range(epochs):
        grads  = NN.computeGradients(train_imgs,steering_angles_matrix)
        params = NN.getParams()   

        params[:] = params[:] - ((lr*grads[:])/(len(train_imgs)))                                       
        NN.setParams(params)

        if iter%100 == 0:
            cost = NN.costFunction(train_imgs[int(iter/100)],steering_angles_matrix[int(iter/100)])
            print('Cost in interation {} = {}'.format(iter, cost))
            if cost < 0.08:
                break
        
        if iter%1000 == 0:
            lr*=5
            

    return NN


# In[19]:


def predict(NN, image_file):
    '''
    Second method you need to complete. 
    Given an image filename, load image, make and return predicted steering angle in degrees. 
    '''
    global min_st_angle, max_st_angle
    bins = np.linspace(min_st_angle,max_st_angle+40, 32)
    
    ## Transform Image and Normalize pixels
    img = cv2.imread(image_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized_img = cv2.resize(gray_image, (32,32))
    crop_img = resized_img[int(resized_img.shape[0]/2):, :]
    crop_img = crop_img/255.
    reshaped_img = np.reshape(crop_img,(1,-1))
    
    yhat = NN.forward(reshaped_img)
    yhat = bins[np.argmax(yhat)]
    
    return yhat


# In[20]:


class NeuralNetwork(object):
    
    def __init__(self):        
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 512
        self.outputLayerSize = 32
        self.hiddenLayerSize = 32
        
        ## Initialize weights properly 
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)*np.sqrt(1/(self.inputLayerSize + self.hiddenLayerSize))
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)*np.sqrt(1/(self.outputLayerSize + self.hiddenLayerSize))
    
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.multiply(np.dot(delta3, self.W2.T),self.sigmoidPrime(self.z2))
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
        
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
        
        
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




