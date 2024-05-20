import torch.nn as nn

def get_conv_layer_names(model):
    conv_layer_names = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            conv_layer_names.append(name)
    return conv_layer_names