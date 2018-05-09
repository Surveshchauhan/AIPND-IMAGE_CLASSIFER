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
import json

#Variables
global modelname
global use_gpu
global json
num_classes = 102
num_hidden = 4096
global use_gpu
global class_to_idx
global img_path
model_weights_path_vgg='data/vgg_checkpoint.pth.tar'
model_weights_path_alex='data/alex_checkpoint.pth.tar'

    
def main():
    args=get_args()
    modelname=args.mo
    use_gpu=args.g
    img_path=args.img
    jsonpath=args.dir
    #prediting the topk classes along with probabilities
    flowerlist,problist=predict(img_path,modelname,5,use_gpu,args.resume)
    print(flowerlist)
    print(problist)
def loadclasses(model):
    (class_to_idx)=model.class_to_idx
    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    for k,v in cat_to_name.items():
        #print(k)
        i=int(k)-1
        class_to_idx[i] = class_to_idx[k]
        del class_to_idx[k]
        class_to_idx[i]=v

    return class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    img_tensor = preprocess(Image.open(image))
    
    return img_tensor
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    # TODO: Process a PIL image for use in a PyTorch model
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loadcheckpoint(model,use_gpu,pathname):
    if model=="vgg16":
        resume_weights = pathname
        print(resume_weights)
        model=vgg(use_gpu)

    else:
        resume_weights = pathname
        model=alex(use_gpu)
    if use_gpu:
        model= model.cuda(0)
    
    if os.path.isfile(resume_weights):
        print("=> loading checkpoint '{}' ...".format(resume_weights))
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
        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights,checkpoint['epoch']))
        return model
def predict(image_path, model, k,use_gpu,pathname):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model_p=loadcheckpoint(model,use_gpu,pathname)
    class_to_idx=loadclasses(model_p)
    if use_gpu:
        data = Variable((process_image(image_path)).unsqueeze_(0), volatile=True).cuda(0)
        model_p = model_p.cuda(0)
    else:
        data = Variable((process_image(image_path)).unsqueeze_(0), volatile=True)
        
    model_p = model_p.eval()
    # apply data to model
    output = model_p(data)

    o=(output.topk(k))
    flowerlist=[]
    problist=[]
    for i in range(5):
        c=(o[1][0][i])
        #print(c[0])
        if use_gpu:
            cin=(c.data.cpu().numpy()[0])
        else:
            cin=(c.data[0])
        flowerlist.append(class_to_idx[cin])
        c=(o[0][0][i])
        if use_gpu:
            cin=(c.data.cpu().numpy()[0])
        else:
            cin=(c.data[0])
       
        problist.append(cin)

    return (flowerlist,problist)

def get_args():
    #retrives and parses the user input via command line
    parser = argparse.ArgumentParser(description='Training a pytorch model to classify different plants')
    parser.add_argument('--mo',  default="vgg16")
    parser.add_argument("--g",  default=False, help='Bool type gpu',action="store_true")
    parser.add_argument('--dir',  help='', default='cat_to_name.json')
    parser.add_argument('--img',  help='', default='/flowers/test/1/image_06743.jpg')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    args = parser.parse_args()
    return args
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
        print()
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
if __name__== "__main__":
    main()
