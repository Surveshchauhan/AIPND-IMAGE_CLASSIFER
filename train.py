# -*- coding: utf-8 -*-
"""
DATE CREATED: 05/08/2018
PURPOSE:
@author: survesh
"""

import argparse
import torch
import bcolz as bz
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import os
import copy

# Hyperparameters
print("Hyperparameters")
global data_dir
batch_size = 64
num_classes = 102
global num_hidden
num_hidden=4096
global learning_rate 
global num_epochs
global use_gpu
global image_datasets
global dataloaders
global dataset_sizes
global class_names
global class_to_idx
best_prec1 = 0
global modelname

model_weights_path_vgg='data/vgg_checkpoint.pth.tar'
model_weights_path_alex='data/alex_checkpoint.pth.tar'


def main():
    args=get_args()
    # Hyperparameters
    data_dir = args.dir
    modelname=args.mo
    
    batch_size = 64
    num_classes = 102
    num_hidden=args.hu
    
    use_gpu=args.g
    
    learning_rate = args.lr
    
    num_epochs = args.ep
    
    #train_dir = data_dir + '/train'
    #valid_dir = data_dir + '/valid'
    #test_dir = data_dir + '/test'
    #if checkpoint available load the saved model of user type, 
    #if not create the model of user type
    image_datasets,dataloaders,dataset_sizes,class_names,class_to_idx=transform(data_dir)    
    if args.resume:
        model=loadcheckpoint(modelname,args.resume,use_gpu)
        print("Testing on saved model")
        if(modelname=="vgg16"):
            conv_feat_train = load_array('data/conv_feat_train.bc')
            labels_train = load_array('data/labels_train.bc')
            conv_feat_val = load_array('data/conv_feat_val.bc')
            labels_val = load_array('data/labels_val.bc')
        else:
            conv_feat_train = load_array('data/conv_feat_trainalex.bc')
            labels_train = load_array('data/labels_trainalex.bc')
            conv_feat_val = load_array('data/conv_feat_valalex.bc')
            labels_val = load_array('data/labels_valalex.bc')
        for param in model.classifier.parameters():
            param.requires_grad = True
        if use_gpu:
            model = model.cuda(0)


        #setting the dropout
        model.classifier[5].p = 0.3
        model.classifier[2].p = 0.3
        if use_gpu:
            model= model.cuda(0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.classifier.parameters(),lr = learning_rate)
        test(conv_feat=conv_feat_val,labels=labels_val,model=model.classifier
                ,size=dataset_sizes['test'],train=False,shuffle=True,use_gpu=use_gpu,criterion=criterion)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        print("=> using pre-trained model '{}'".format(modelname))
        model = models.__dict__[modelname](pretrained=True)
        num_ftrs = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1]
        features.extend([nn.Linear(num_ftrs, num_classes)])
        model.classifier = nn.Sequential(*features)
        #creating a classfier 
        model = FineTuneModel(model, modelname, num_classes,num_hidden=4096)
        image_datasets,dataloaders,dataset_sizes,class_names,class_to_idx=transform(data_dir)    
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6].out_features = num_classes

        #freezing the layers
        for param in model.classifier[6].parameters():
            param.requires_grad = True
        if use_gpu:
            
            model = model.cuda(0)
        print("getting the features")
        conv_feat_train,labels_train = preconvfeat(dataloaders['train'],model,use_gpu)
        conv_feat_val,labels_val = preconvfeat(dataloaders['val'],model,use_gpu)

        #conv_feat_train = load_array('data/conv_feat_trainalex.bc')
        #labels_train = load_array('data/labels_trainalex.bc')
        #conv_feat_val = load_array('data/conv_feat_valalex.bc')
        #labels_val = load_array('data/labels_valalex.bc')
        if(modelname=="vgg16"):
            save_array('data/conv_feat_train.bc', conv_feat_train)
            save_array('data/labels_train.bc', labels_train)
            save_array('data/conv_feat_val.bc', conv_feat_val)
            save_array('data/labels_val.bc', labels_val)
        else:
            save_array('data/conv_feat_trainalex.bc', conv_feat_train)
            save_array('data/labels_trainalex.bc', labels_train)
            save_array('data/conv_feat_valalex.bc', conv_feat_val)
            save_array('data/labels_valalex.bc', labels_val)
        #Traing the last layer
        for param in model.classifier.parameters():
            param.requires_grad = True
        if use_gpu:
            model = model.cuda(0)


        #setting the dropout
        model.classifier[5].p = 0.3
        model.classifier[2].p = 0.3
        if use_gpu:
            model= model.cuda(0)
          # define loss function (criterion) and optimizer    
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.classifier.parameters(),lr = learning_rate)
        #train the network and save_checkpoint is used int he training itself to save the best model along with class_to_idx
        print("Training" )
        if(modelname=="vgg16"):
            train(model=model.classifier,size=dataset_sizes,train_b=conv_feat_train,train_labels=labels_train,val_b=conv_feat_val,val_lables=labels_val,
                epochs=num_epochs,optimizer=optimizer,criterion=criterion,train=True,shuffle=True,filename=model_weights_path_vgg,use_gpu=use_gpu,class_to_idx=class_to_idx)
            print("Validating on Test Data")
            test(conv_feat=conv_feat_val,labels=labels_val,model=model.classifier
                ,size=dataset_sizes['test'],train=False,shuffle=True,use_gpu=use_gpu,criterion=criterion)
        else :
                train(model=model.classifier,size=dataset_sizes,train_b=conv_feat_train,train_labels=labels_train,val_b=conv_feat_val,val_lables=labels_val,
                epochs=num_epochs,optimizer=optimizer,criterion=criterion,train=True,shuffle=True,filename=model_weights_path_alex,use_gpu=use_gpu,class_to_idx=class_to_idx)
        print("Validating on Test Data")
        test(conv_feat=conv_feat_val,labels=labels_val,model=model.classifier
                ,size=dataset_sizes['test'],train=False,shuffle=True,use_gpu=use_gpu,criterion=criterion)

        
        #saving the model abd associated to check point
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loadcheckpoint(model,path,use_gpu):
    if model=="vgg16":
        resume_weights = path
        model=vgg(use_gpu)

    else:
        resume_weights = path
        model=alex(use_gpu)
    if use_gpu:
        model= model.cuda(0)
    
    if os.path.isfile(resume_weights):
        #print("=> loading checkpoint '{}' ...".format(resume_weights))
        if use_gpu:
            checkpoint = torch.load(resume_weights)
        else:
            # Load GPU model on CPU
            checkpoint = torch.load(resume_weights,
                                    map_location=lambda storage,
                                    loc: storage)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.classifier.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx=(checkpoint['class_to_idx'])
        print("=> loaded checkpoint '{}'".format(resume_weights))
        return model
# TODO: Save the checkpoint 
# Saving The Model:

def transform(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    #data_transforms = 
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    }
    # TODO: Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val','test']}
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    
    class_names = image_datasets['train'].classes
    class_to_idx = image_datasets['train'].class_to_idx
    return (image_datasets,dataloaders,dataset_sizes,class_names,class_to_idx)
    #use_gpu = torch.cuda.is_available()

def get_args():
    #retrives and parses the user input via command line

    parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
    parser.add_argument('--dir',  help='', default='flowers')
    parser.add_argument('--mo',  default="vgg16")
    parser.add_argument('--ep',  default=100, type=int)
    parser.add_argument('--hu',  default=4096,type=int)
    parser.add_argument('--lr',  default=4096,type=float)
    parser.add_argument("--g",   default=False, action="store_true", help='Bool type gpu')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args

def train(model,size,train_b=None,train_labels=None,val_b=None,val_lables=None,epochs=1,optimizer=None,criterion=None,train=True,shuffle=True,filename=None,use_gpu=True,class_to_idx=None):
    best_accuracy=0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs -1))
        
        for phase in ['train', 'val']:
            total = 0
            running_loss = 0.0
            running_corrects = 0
            if phase == 'train':
                model.train()
                batches = data_gen(conv_feat=train_b,labels=train_labels,shuffle=shuffle)
            
                for inputs,classes in batches:
                    if use_gpu:
                        inputs , classes = Variable(torch.from_numpy(inputs).cuda(0)),Variable(torch.from_numpy(classes).cuda(0))
                    else:
                        inputs, labels = Variable(torch.from_numpy(inputs)), Variable(torch.from_numpy(labels))
                
                    inputs = inputs.view(inputs.size(0), -1)
                    outputs = model(inputs)
                    loss = criterion(outputs,classes)           
                    if train:
                        if optimizer is None:
                            raise ValueError('Pass optimizer for train mode')
                    optimizer = optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    _,preds = torch.max(outputs.data,1)
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == classes.data)
            else:
                model.eval()
                batches = data_gen(conv_feat=val_b,labels=val_lables,shuffle=shuffle)
                for inputs,classes in batches:
                    if use_gpu:
                        inputs , classes = Variable(torch.from_numpy(inputs).cuda(0)),Variable(torch.from_numpy(classes).cuda(0))
                    else:
                        inputs, labels = Variable(torch.from_numpy(inputs)), Variable(torch.from_numpy(labels))
                
                    inputs = inputs.view(inputs.size(0), -1)
                    outputs = model(inputs)
                    loss = criterion(outputs,classes)           
                    
                    _,preds = torch.max(outputs.data,1)
                    # statistics
                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == classes.data)
                
            epoch_loss = running_loss / size[phase]
            epoch_acc = running_corrects / size[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_accuracy:
                is_best = epoch_acc > best_accuracy
                best_accuracy = epoch_acc
                model.class_to_idx=class_to_idx
                state = {
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_accuracy': best_accuracy,
                        'class_to_idx':model.class_to_idx
                        }
                save_checkpoint(state, is_best,filename)
    print()
def test(model,size,conv_feat=None,labels=None,epochs=1,optimizer=None,train=True,shuffle=True,use_gpu=True,criterion=None):
    for epoch in range(epochs):
        batches = data_gen(conv_feat=conv_feat,labels=labels,shuffle=shuffle)
        total = 0
        running_loss = 0.0
        running_corrects = 0
        for inputs,classes in batches:
            if use_gpu:
                inputs , classes = Variable(torch.from_numpy(inputs).cuda(0)),Variable(torch.from_numpy(classes).cuda(0))
            else:
                inputs, labels = Variable(torch.from_numpy(inputs)), Variable(torch.from_numpy(labels))
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs,classes)           
            if train:
                if optimizer is None:
                    raise ValueError('Pass optimizer for train mode')
                optimizer = optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            _,preds = torch.max(outputs.data,1)
            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == classes.data)
        epoch_loss = running_loss / size
        epoch_acc = running_corrects / size
        print('Loss: {:.4f} Acc: {:.4f}'.format(
                     epoch_loss, epoch_acc))
def data_gen(conv_feat,labels,batch_size=batch_size,shuffle=True):
    labels = np.array(labels)
    if shuffle:
        index = np.random.permutation(len(conv_feat))
        conv_feat = conv_feat[index]
        labels = labels[index]
    for idx in range(0,len(conv_feat),batch_size):
        yield(conv_feat[idx:idx+batch_size],labels[idx:idx+batch_size])

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

def save_array(filename, arr):
    c=bz.carray(arr, rootdir=filename, mode='w')
    c.flush()
def load_array(filename):
    return bz.open(filename)[:]

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes,num_hidden):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, num_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(num_hidden, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(num_features, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, num_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(num_hidden, num_classes),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

#Loading the vgg16 model and freezing the parameters for feature network
def vgg(use_gpu):
    model_v= torchvision.models.vgg16(pretrained=True)
    num_ftrs = model_v.classifier[6].in_features
    features = list(model_v.classifier.children())[:-1]
    features.extend([nn.Linear(num_ftrs, num_classes)])
    model_v.classifier = nn.Sequential(*features)
    model_v = FineTuneModel(model_v, 'vgg16', num_classes,num_hidden)
    #print(model)
    for param in model_v.parameters():
        param.requires_grad = False
    model_v.classifier[6].out_features = num_classes
    for param in model_v.classifier[6].parameters():
        param.requires_grad = True
    if use_gpu:
        model_v = model_v.cuda(0)
    #model_v.class_to_idx = class_to_idx
    return model_v

#Replicating the work done on vgg for alexnet
def alex(use_gpu):
    model_a = torchvision.models.alexnet(pretrained=True)
    num_ftrs = model_a.classifier[6].in_features
    features = list(model_a.classifier.children())[:-1]
    features.extend([nn.Linear(num_ftrs, num_classes)])
    model_a.classifier = nn.Sequential(*features)
    model_a = FineTuneModel(model_a, "alexnet", num_classes,num_hidden)
    #print(model)
    for param in model_a.parameters():
        param.requires_grad = False
    model_a.classifier[6].out_features = num_classes
    for param in model_a.classifier[6].parameters():
        param.requires_grad = True
    if use_gpu:
        model_a = model_a.cuda(0)
    #model_a.class_to_idx = class_to_idx
    return model_a
def preconvfeat(dataset,model,use_gpu):
    conv_features = []
    labels_list = []
    
    for data in dataset:
        inputs,labels = data
        if use_gpu:
            inputs , labels = Variable(inputs.cuda(0)),Variable(labels.cuda(0))
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        x = model.features(inputs)
        conv_features.extend(x.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())
    conv_features = np.concatenate([[feat] for feat in conv_features])
    return (conv_features,labels_list)

if __name__== "__main__":
    main()