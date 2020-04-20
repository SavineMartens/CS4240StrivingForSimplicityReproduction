lr_values = [0.25, 0.1, 0.05, 0.01] 
num_epochs = 350
weight_decay = 0.001

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import os
#import ModelsA as mod
import torch.nn as nn
import torch.nn.functional as F
import datetime
verbal = False

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Loading and preprocessing data
dataset_path = './cifar10'
trainset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
train_mean = trainset.data.mean(axis=(0,1,2))/255  # I stole this from the author
train_std = trainset.data.std(axis=(0,1,2))/255  
testset = datasets.CIFAR10(root=dataset_path, train=True, download=True)
test_mean = trainset.data.mean(axis=(0,1,2))/255  
test_std = trainset.data.std(axis=(0,1,2))/255  
batch_size = 64

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(train_mean, train_std)])  

transform_test = transforms.Compose(
     [transforms.ToTensor(),
      transforms.Normalize(test_mean, test_std)])  

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,
                                          pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0,
                                         pin_memory=True)
labels = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Trainer():
    def __init__(self,modelRun):
      self.modelRun=modelRun
    def train(self,model, lr_val,  globaliter, device):  
      pytorch_total_params = sum(p.numel() for p in model.parameters())
      print("parameter count")
      print(pytorch_total_params)
      model.train()
      criterion = nn.CrossEntropyLoss() # article doesn't say which loss function it's using
      running_loss = 0.0
      # print (len(trainloader))
      print('Total number of batches:', len(trainloader))
      for i, data in enumerate(trainloader):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.to(device)
          labels = labels.to(device)
          globaliter += 1 

          # zero the parameter gradients
          optimizer = optim.SGD(model.parameters(), lr=lr_val, momentum=0.9)  # weight_decay?
          optimizer.zero_grad()

          # TRAINING: forward + backward + optimize
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 200 == 199 and verbal == False:    # print every 200 mini-batches
              print('[%d] loss: %.3f' %(i + 1, running_loss / 200))
              running_loss = 0.0
              # niter = epoch * len(trainloader) + i
              # writer.add_scalar('Train/loss',loss.item(),niter)
          if verbal: 
            log('batch', i+1, 'loss', loss.item())
          # This is where I'm recording to Tensorboard
          # tb.save_value('Train Loss', 'train_loss', globaliter, loss.item())

          # # VALIDATION
          #     outputs = net(inputs)
          #     _, predicted = torch.max(outputs.data, 1)
          #     total += labels.size(0)
          #     correct += (predicted == labels).sum().item()

          # if i == len(trainloader)-1:
          #   print('Accuracy of epoch on 10000 validation images: %d %%' % (
          #   100 * correct / total))

      return loss.item(), model.state_dict(), model, globaliter


    def test_val(self,model, save_path, best_loss, model_dict, device):
      model.eval()
      criterion = nn.CrossEntropyLoss() # article doesn't say which loss function it's using
      loss_all = []
      total = 0
      correct = 0
      print('Total number of test batches:', len(testloader))

      with torch.no_grad():
        for i, data in enumerate(testloader):
            if verbal:
                print('Test batch', i+1)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            if i == len(testloader)-1:
                accuracy =  100 * correct / total
                print('Accuracy of epoch on 10000 validation images: %d %%' % (
            100 * correct / total))
            loss_all.append(loss.item())

        avg_loss = sum(loss_all)/len(loss_all)

      return avg_loss, accuracy





    ##### TRAINING


    def modelTrain(self,save_path, lr_val_original):
        
        net = self.modelRun
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        net = net.to(device)
            
        epochs_to_update = [199, 249, 299]

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        string_file = save_path + "/" + str(lr_val_original) + "TrainingProcess.txt"
        f= open(string_file,"w+")
        f.write("Epoch ValidationLoss Accuracy\n")
        f.close()

        # for ii, lr_val in enumerate(lr_values):
        t_0 = time.time()
        best_loss = 1000
        lr_val = lr_val_original
        globaliter = 0
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            if epoch == 1:
                t_epoch = time.time() - t_0
                print('Training time per epoch:', t_epoch)
            # learning rate updating during training
            if epoch in epochs_to_update:
                lr_val = lr_val*0.1
            print('learning rate:', lr_val, 'Epoch', epoch+1) 

            loss_epoch, model_dict, model, globaliter = self.train(net, lr_val, globaliter, device)
            avg_loss, accuracy = self.test_val(net, save_path, best_loss, model_dict, device)

            f= open(string_file,"a+")
            f.write(str(epoch+1) + ' ' + str(avg_loss) + ' ' + str(accuracy)  + "\n")
            f.close()

            if avg_loss < best_loss:
              best_model_dict = model_dict
              torch.save(best_model_dict, save_path + '/LR' + str(lr_val_original) 
                         + '.pt')   
              best_loss = avg_loss
              best_acc = accuracy
              print('Saving new model with best loss', best_loss)
              

        total_training_time = time.time() - t_0
        print('Total training time:', total_training_time)
        print('Best accuracy:', best_acc)

