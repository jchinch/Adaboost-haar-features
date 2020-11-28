#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:38:36 2020

@author: jatinchinchkar
"""
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from time import time
from numpy import matlib
from tqdm import tqdm

root_path = "/Users/jatinchinchkar/Desktop/CV/Project2/"
face_path = root_path+"Face16/"
non_face_path = root_path+"Nonface16/"

train_face_size = 100
train_nface_size = 100
test_size = 100

temp_face = os.listdir(face_path)
temp_nface = os.listdir(non_face_path)
imsize = (10,10)
###################### Creating the dataset for face, non face, test and train data ##############################
def getData(path,ls,size_train,size_test):
    train_data, test_data = [],[]
    for i in tqdm(range(size_train)):
        img1 = np.array((Image.open(path+str(ls[i]))).resize(imsize,Image.BILINEAR))
        #temp1 = np.reshape(img1,(-1,1))/255
        temp1 = img1/255
        train_data.append(temp1)
    for i in tqdm(range(size_train,size_train+size_test)):
        img1 = np.array((Image.open(path+str(ls[i]))).resize(imsize,Image.BILINEAR))
        #temp1 = np.reshape(img1,(-1,1))/255
        temp1 = img1/255
        test_data.append(temp1)
    return train_data,test_data
###################### Creating the integral images for eavery image ###########################################
def integralImage(ls):
    for i in tqdm(range(len(ls))):
        ls[i] = ls[i].cumsum(axis=1).cumsum(axis=0)
    return ls
###################### Function to check if the feature is a possible for given image ##########################
def possibleFeat(i,x,y,x_,y_):
    featureType = [[1,2],[2,1],[1,3],[3,1],[2,2]]
    end_x = featureType[i][1]*x_
    end_y = featureType[i][0]*y_
    if (x+end_x<10) and (y+end_y<10):
        possiFeat = True
    else:
        possiFeat = False
    return possiFeat
###################### 5 haar like features from image ###############################
def featureExtraction(img,i,x,y,x_,y_):
    featureType = [[1,2],[2,1],[1,3],[3,1],[2,2]]
    end_x = x + featureType[i][1] * x_ - 1
    end_y = y + featureType[i][0] * y_ - 1
    # print('endx,endy',end_x,end_y)
    # print('x,y',i,x,y,x_,y_)
    if i == 0:
        start_col,start_row = x,y
        end_row = end_y
        end_col = (end_x+x)//2
        left_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_col = (end_x+x)//2
        start_row = y
        end_col = end_x 
        end_row = end_y
        right_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][start_col]
        feature = right_area - left_area
    
    elif i == 1:  
        start_col,start_row = x,y
        end_row = (end_y+y)//2
        end_col = end_x
        top_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_row = (end_y+y)//2
        end_col = end_x 
        end_row = end_y
        bot_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][start_col]
        feature = top_area - bot_area
        
    elif i == 2:
        start_col,start_row = x,y
        end_col = x+(end_x-x)//3
        end_row = end_y
        left_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_col = end_col
        end_col = x+2*(end_x-x)//3
        mid_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_col = end_col
        end_col = end_x
        right_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        feature = mid_area - (left_area+right_area)
        
    elif i == 3:
        start_col,start_row = x,y
        end_col = end_x
        end_row = y+(end_y-y)//3
        top_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_row = end_row
        end_row = y+2*(end_y-y)//3
        mid_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_row = end_row
        end_row = end_y
        bot_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        feature = mid_area - (top_area+bot_area)
        
    else:
        start_col,start_row = x,y
        end_row = (end_y+y)//2
        end_col = (end_x+x)//2
        first_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_col = end_col
        start_row = y
        end_col = end_x
        end_row = end_y
        second_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_row = (end_y+y)//2
        start_col = x
        end_col = (end_x+x)//2
        end_row = end_y
        third_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        start_col = (end_x+x)//2
        start_row = (end_y+y)//2
        end_row = end_y
        end_col = end_x
        fourth_area = img[end_row][end_col]+img[start_row][start_col]-img[start_row][end_col]-img[end_row][end_col]
        feature = (third_area+second_area)-(first_area+fourth_area)
        
    return feature
###################### Extraction of all possibile haar like features from image ###############################    
def getfeature(img):
    ls = []
    count = 0
    for i in tqdm(range(5)):
        for x_ in range(10):
            for y_ in range(10):
                for x in range(10):
                    for y in range(10):
                        if ((possibleFeat(i,x,y,x_,y_)) == True):
                            temp = featureExtraction(img,i,x,y,x_,y_)
                            ls.append([temp,i,x_,y_,x,y,count])
                            count += 1
    return ls
###################### Finding threshold value for everu haar feature ###############################
def threshold(face,nface):
    ls = np.concatenate((face,nface))
    min_val = ls.min()
    max_val = ls.max()
    max_count,thresh = 0,0
    for j in tqdm(np.arange(min_val,max_val,((max_val-min_val)/90))):
        countf,countnf = 0,0
        temp_face = np.where(j<face,1,0)
        temp_nface = np.where(j>nface,1,0)
        countf = np.count_nonzero(temp_face==1)
        countnf = np.count_nonzero(temp_nface==1)
        if (countf+countnf) >= max_count:
            max_count = (countf+countnf)
            thresh = j
    return max_count,thresh

def getFeat(ls):
    features = []
    for img in tqdm(ls):
        temp = getfeature(img)
        temp = np.array(temp)
        features.append(temp)
    return features

def listfeatures(ls):
    feat = []
    ls = np.array(ls)
    for i in tqdm(range(ls.shape[0])):
        img = []
        for j in range(ls.shape[1]):
            img.append(np.array(ls[i][j][0]))
        feat.append(img)  
    return feat
###################### Implementing Adaboost algorithm to get top 10 weak classifiers ###############################
def adaboost(strong_value,face_features,nface_features,t):
    y = np.concatenate((np.ones(shape=[1,len(face_features)]),np.negative(np.ones(shape=[1,len(face_features)]))),axis=1)
    h = np.zeros(shape=[t,y.shape[1]])
    wts = np.matlib.repmat(1/y.shape[1],1,y.shape[1])
    alphaT,ht = [],[]
    Ht = np.zeros(shape=[1,y.shape[1]])
    for i in tqdm(range(t)):
        error = []
        for j in range(strong_value.shape[0]):
            for k in range(len(face_features)):
                if strong_value[j][0] < face_features[k][j][0]:
                    h[i][k] = 1
                else:
                    h[i][k] = -1
            for k in range(len(face_features)):
                if strong_value[j][0] > nface_features[k][j][0]:
                    h[i][len(face_features)+k] = -1
                else:
                    h[i][len(face_features)+k] = 1
            err = 0
            for k in range(y.shape[1]):
                if y[0][k] != h[i][k]:
                    err += wts[0][k]
            error.append(err)
        feature = np.argmin(error)
        et = error[feature]
        alpha = 0.5*(np.log((1-et)/(et)))
        alphaT.append(alpha)
        htemp = np.zeros(shape=[1,y.shape[1]])
        for k in range(len(face_features)):
            if strong_value[feature][0] < face_features[k][feature][0]:
                htemp[0][k] = 1
            else:
                htemp[0][k] = -1
        for k in range(len(face_features)):
            if strong_value[feature][0] > nface_features[k][feature][0]:
                htemp[0][len(face_features)+k] = -1
            else:
                htemp[0][len(face_features)+k] = 1
        z = 0
        for k in range(len(wts[0])):
            z = z + wts[0][k]*np.exp(-1*y[0][k]*alpha*htemp[0][k]) 
        for k in range(len(wts[0])):
            wts[0][k] = (wts[0][k]*np.exp(-1*y[0][k]*alpha*htemp[0][k]))/z
        ht.append([strong_value[feature][0],feature,alpha,htemp])
        Ht = Ht + np.multiply(ht[i][2],ht[i][3])[0]
        accuracy = np.count_nonzero(np.equal(np.sign(Ht),y))/y.shape[1]
        print('Training Accuracy for iteraion is',accuracy)

    return ht
###################### ROC generation for model ###############################
def regressor_to_classifier(predictions, threshold=0.5):
    output = []
    for prediction in predictions:
        if prediction > threshold: 
            output.append(1)
        else: 
            output.append(-1)
    return output

def confusion_matrix(true, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for t, p in zip(true, predictions):
        #print(t,p)
        if t == 1 and p == 1: 
            TP += 1
        elif t == -1 and p == 1:
            FP += 1
        elif t == 1 and p == -1:  
            FN += 1
        else: 
            TN += 1
    return TP, FP, TN, FN

def roc_curve(true,h_final,alphas):
    x = []
    y = []
    for i in range(100):
        threshold = 0.01 * i
        float_predictions = np.sign(np.sum((alphas*(h_final-threshold)),axis=0))
        #print(type(float_predictions[0]))
        bool_predictions = regressor_to_classifier(float_predictions, threshold)
        TP, FP, TN, FN = confusion_matrix(true, bool_predictions)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        x.append(FPR)
        y.append(TPR)
    plt.plot(x, y)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()    
###################### Extraction of all possibile haar like features from test image ###############################    
def getReqTestFearures(ht,face_test_features,nface_test_features,t):
    top_ten_value = np.take(ht,1,axis=1)
    req_tface_features = np.zeros(shape=[t,len(test_face)])
    req_tnface_features = np.zeros(shape=[t,len(test_face)])
    for i in tqdm(range(len(top_ten_value))):
        idx = int(top_ten_value[i])
        for j in range(len(test_face)):
            req_tface_features[i][j] = face_test_features[j][idx][0]
            req_tnface_features[i][j] = nface_test_features[j][idx][0]
    return req_tface_features,req_tnface_features
###################### Testing the classifier on test images ###############################    
def testModel(ht,req_tface_features,req_tnface_features):
    h_final = np.zeros(shape=[t,2*test_size])
    for i in tqdm(range(t)):
        for j in range(t):
            for k in range(len(req_tface_features[0])):
                if ht[j][0] < req_tface_features[j][k]:
                    h_final[j][k] = 1
                else:
                    h_final[j][k] = -1
            for k in range(len(req_tface_features[0])):
                if ht[j][0] > req_tnface_features[j][k]:
                    h_final[j][k+len(req_tface_features[0])] = -1
                else:
                    h_final[j][k+len(req_tface_features[0])] = 1 
    ytest = np.concatenate((np.ones(shape=[1,test_size]), np.negative(np.ones(shape=[1,test_size]))),axis=1)
    alphas = np.reshape((np.take(ht,2,axis=1)),[-1,1])
    output = np.sum((alphas*h_final),axis=0)
    accuracy_test = np.count_nonzero(np.equal(np.sign(output),ytest))/(2*test_size)
    print('Test Accuracy of the model is',accuracy_test)
    roc_curve(ytest[0].T, h_final, alphas)
    return output
###################### Fetching Threshold value for all haar like features ###############################    
def getGetThreshold(ls_face_feat,ls_nface_feat,face_features):
    strong_value = []
    for j in tqdm(range(len(ls_face_feat[0]))):
        temp1 = np.take(ls_face_feat,j,axis=1)
        temp2 = np.take(ls_nface_feat,j,axis=1)
        thresh,feature_val = threshold(temp1, temp2)
        temp_arr = face_features[0][j]
        temp_arr[0] = feature_val
        temp = np.insert(temp_arr,1,thresh)
        strong_value.append(temp)
    return strong_value
###################### Visualization of top ten haar like features ############################### 
def drawFeatures(img,ls):
    i = int(ls[2])
    x = int(ls[5])
    y = int(ls[6])
    x_ = int(ls[3])
    y_ = int(ls[4])
    featureType = [[1,2],[2,1],[1,3],[3,1],[2,2]]
    end_x = x + featureType[i][1] * x_ - 1
    end_y = y + featureType[i][0] * y_ - 1
    visual = []
    if i == 0:
        start_col,start_row = x,y
        end_row = (end_y)
        end_col = (end_x+x)//2
        i = cv2.rectangle(img,(start_row,start_col),(end_row,end_col),(255))
        start_col = end_col
        end_col = end_x
        end_row = end_y
        image = cv2.rectangle(i,(start_row,start_col),(end_row,end_col),(0))
    elif i == 1:
        start_col,start_row = x,y
        end_col = end_x
        end_row = (end_y+y)//2
        i = cv2.rectangle(img,(start_row,start_col),(end_row,end_col),(255))
        start_row = end_row
        end_col = end_x
        end_row = end_y
        image = cv2.rectangle(i,(start_row,start_col),(end_row,end_col),(0))
    elif i == 2:
        start_col,start_row = x,y
        end_row = end_y
        end_col = x+(end_x-x)//3
        i = cv2.rectangle(img,(start_row,start_col),(end_row,end_col),(255))
        start_col = end_col
        start_row = y
        end_col = x + 2*(end_x-x)//3
        i1 = cv2.rectangle(i,(start_row,start_col),(end_y,end_col),(0))
        start_col = end_col
        end_col = end_x
        end_row = end_y
        image = cv2.rectangle(i1,(start_row,start_col),(end_y,end_x),(255))
    elif i == 3:
        start_col,start_row = x,y
        end_row = y +(end_y-y)//3
        end_col = end_x
        i = cv2.rectangle(img,(start_row,start_col),(end_row,end_col),(255))
        start_row = y + (end_y-y)//3
        end_row = y + 2*(end_y-y)//3
        i1 = cv2.rectangle(i,(start_row,start_col),(end_y,end_x),(0))
        start_row = end_row
        end_row = end_y
        image = cv2.rectangle(i1,(start_row,start_col),(end_y,end_x),(255))
    else:
        start_col,start_row = x,y
        end_col = (end_x+x)//2
        end_row = (end_y+y)//2
        i = cv2.rectangle(img,(start_row,start_col),(end_row,end_col),(255))
        start_col = end_col
        end_col = end_x
        i1 = cv2.rectangle(i,(start_row,start_col),(end_row,end_col),(0))
        start_row = end_row
        end_row = end_y
        start_col = x
        end_col = (end_x+x)//2
        i2 = cv2.rectangle(i1,(start_row,start_col),(end_row,end_col),(255))
        start_col = (end_x+x)//2
        start_row = (end_y+y)//2
        image = cv2.rectangle(i2,(start_row,start_col),(end_y,end_x),(0))
    return image
 
def visualizeHar(ht,strong_value,t):   
    top_ten_value = np.take(ht,1,axis=1)
    feature_info =[]
    for j in range(strong_value.shape[0]):
        if j in top_ten_value:
            feature_info.append(strong_value[j])
    visual = []
    for j in range(t):
        img1 = np.array(Image.open(face_path+'c000001.BMP'))
        temp = drawFeatures(img1, feature_info[j])       
        visual.append(temp)  

print('start')     
t = 10
t_start = time()   
train_face_data,test_face_data = getData(face_path, temp_face, train_face_size, test_size)
train_nface_data,test_nface_data = getData(non_face_path, temp_nface, train_nface_size, test_size)
train_face = integralImage(train_face_data)
train_nface = integralImage(train_nface_data)
test_face = integralImage(test_face_data)
test_nface = integralImage(test_nface_data)
face_features = getFeat(train_face)
nface_features = getFeat(train_nface)
face_test_features = getFeat(test_face)
nface_test_features = getFeat(test_nface)
ls_face_feat = listfeatures(face_features)
ls_nface_feat = listfeatures(nface_features)
ls_face_feat = np.array(ls_face_feat)
ls_nface_feat = np.array(ls_nface_feat)
strong_value = getGetThreshold(ls_face_feat, ls_nface_feat, face_features)
face_features = getFeat(train_face)
ls_face_feat = listfeatures(face_features)
strong_value = np.array(strong_value)
print('threshold calculated')
ht = adaboost(strong_value, face_features, nface_features, t)
req_tface_features,req_tnface_features = getReqTestFearures(ht, face_test_features, nface_test_features, t)
output = testModel(ht, req_tface_features, req_tnface_features)
visualizeHar(ht, strong_value, t)
t_stop = time()
print('Done')
print('execution Time',t_stop-t_start)

               
