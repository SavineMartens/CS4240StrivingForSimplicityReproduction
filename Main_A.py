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

if model_to_train == 1:
    from ModelsA import Model_A

    trainVar = Trainer(Model_A())
    trainVar.modelTrain('./Model_A', lr_rate)

if model_to_train == 2:
    from ModelsA import Model_Strided_A

    trainVar = Trainer(Model_Strided_A())
    trainVar.modelTrain('./Model_Strided_A', lr_rate)

if model_to_train == 3:
    from ModelsA import Model_ConvPool_A

    trainVar = Trainer(Model_ConvPool_A())
    trainVar.modelTrain('./Model_ConvPool_A', lr_rate)

if model_to_train == 4:
    from ModelsA import Model_All_A

    trainVar = Trainer(Model_All_A())
    trainVar.modelTrain('./Model_All_A', lr_rate)
