#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import scipy.io as sio
import time
import gc
import sklearn.metrics
from sklearn.metrics import classification_report,f1_score
from sklearn.metrics import accuracy_score
import ABIDEParser as Reader
import glob
from utils import *
from model import *
from cal_weight import *


subject_IDs = np.genfromtxt("subject_IDs.txt", dtype=str)
subject_IDs = subject_IDs.tolist()

print(len(subject_IDs))
print(subject_IDs[:5])

numClass = 1

# Get subject labels
label_dict = Reader.get_label(subject_IDs)
label_list = np.array([int(label_dict[x]) for x in subject_IDs])
print(label_list[:5])



train_fname = 'train_sub.txt'
test_fname = 'test_sub.txt'
sub_train = np.loadtxt(train_fname, dtype=str)
sub_test = np.loadtxt(test_fname, dtype=str)

print('train shape:',sub_train.shape,'test shape:',sub_test.shape)

data_test, y_test = Reader.getconn_vector(sub_test, 'correlation', 'aal', label_dict)
    
x_test = np.arctanh(data_test)
    
xdim = x_test.shape[1]
model = FCNN_model(xdim, numClass)

bestModelSavePath ='FCNN_aal.hdf5'
    
# test
model.load_weights(bestModelSavePath)
print('Start Predict.')
pred = model.predict(x_test)
print("pred shape:",pred.shape, "y_true shape", y_test.shape)

y_pred = []
for p in pred:
    y_pred.append(round(p[0]))
y_pred = np.array(y_pred)
    
result = Reader.evaluate(pred, y_pred, y_test)
SS = result[0]
GR = result[2]
print(result)

fx = get_weight(model, x_test)
print(fx.shape)



#top K weight analyze
all_avg_CPP = []
all_NLCI = []
sensitivity = [SS]
accs = [GR]
select_count = []
before_ind = []
imp_ind = []
for k in range(5, 305, 5):
    print("##############", k, "############")
    all_topK_index = get_topK_feaInd(fx, k)
    print(np.array(all_topK_index).shape)
    feaInd_dict = get_ind_freq(all_topK_index)
    print(feaInd_dict)
    select_feaInd = select_index(feaInd_dict, len(y_test), 0.95)
    print(select_feaInd)
    
    diff_ind = [i for i in select_feaInd if i not in before_ind]
    imp_ind.extend(diff_ind)
    before_ind = imp_ind
    select_count.append(len(imp_ind))
    print("select impInd length:", len(imp_ind))
    
    hack_pred, label_pred, hack_SS, hack_acc = hack_model(model, x_test, y_test, imp_ind)
    CPP, avg_CPP = cal_CPP(pred, hack_pred)
    NLCI = cal_NLCI(y_pred, label_pred)
    all_avg_CPP.append(avg_CPP)
    all_NLCI.append(NLCI)
    sensitivity.append(hack_SS)
    accs.append(hack_acc)

print(all_avg_CPP)
print(all_NLCI)






