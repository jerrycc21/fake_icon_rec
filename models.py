import os
import torch
import torchvision
import torch.utils.model_zoo as model_zoo

def get_model(load_pretrain=True):
    model_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    model = torchvision.models.resnet18()
    if load_pretrain:
        model.load_state_dict(model_zoo.load_url(model_url))
    model.fc = torch.nn.Linear(512, 2)
    return model
