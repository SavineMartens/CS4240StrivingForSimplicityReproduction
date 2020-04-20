# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 23:38:13 2020

@author: Savine
"""
import sys


model_to_train = int(sys.argv[1])
lr_rate = float(sys.argv[2])

from Trainer import Trainer
lr_values = [0.25, 0.1, 0.05, 0.01] 

if lr_rate not in lr_values:
    sys.exit('Learning rate not in possible values!')

## Training Model_C WORKS!
if model_to_train == 1:
    from ModelsC import Model_C
    trainVar=Trainer(Model_C())
    trainVar.modelTrain('./Model_C', lr_rate)


## Training Model_Strided_C DOESN NOT WORK
if model_to_train == 2:
    from ModelsC import Model_Strided_C
    trainVar=Trainer(Model_Strided_C())
    trainVar.modelTrain('./Model_Strided_C', lr_rate)


# Training Model_ConvPool_C WORKS! 0.25, 0.1 stopped prematurely
if model_to_train == 3:
    from ModelsC import Model_ConvPool_C
    trainVar=Trainer(Model_ConvPool_C())
    trainVar.modelTrain('./Model_ConvPool_C', lr_rate)


# Training Model_All_C 
if model_to_train == 4:
    from ModelsC import Model_All_C
    trainVar=Trainer(Model_All_C())
    trainVar.modelTrain('./Model_All_C', lr_rate)
    
    
