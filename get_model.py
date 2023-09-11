import torch
import torch.nn as nn
from ResNets_3D_CNN.models import resnet

def get_model(pretrained_weight_path: str, minimum_order: int, device: torch.device):

    CNN3D_model = resnet.generate_model(model_depth = 18, n_classes = 1039)
    pretrain = torch.load(pretrained_weight_path, map_location='cpu')
    CNN3D_model.load_state_dict(pretrain['state_dict'])
    CNN3D_model.fc = torch.nn.Sequential(
                        nn.Linear(in_features = 512, out_features = 128),
                        nn.ReLU(),
                        nn.Linear(in_features = 128, out_features = 9)
                    )

    CNN3D_model = CNN3D_model.to(device)

    for param in CNN3D_model.parameters():
            param.requires_grad = False

    for order,child in enumerate(CNN3D_model.children()):
        if order > minimum_order:
            for param in child.parameters():
                param.requires_grad = True

    return CNN3D_model