import sys
sys.path.append('../../U-Net')

import torch.nn as nn
import e2cnn.nn as e2_nn
from models import *
from building_blocks import *



def getModel(model_name, settings):

    if settings['models'][model_name]['acti'] == 'relu':
        if model_name == 'UNet_vanilla' or model_name == 'UNet_bcnn':
            acti = nn.ReLU
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
        elif model_name == 'UNet_e2cnn':
            acti = e2_nn.ReLU
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
    elif settings['models'][model_name]['acti'] == 'tanh':
        if model_name == 'UNet_vanilla' or model_name == 'UNet_bcnn':
            acti = nn.Tanh
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
        elif model_name == 'UNet_e2cnn':
            raise NotImplementedError("tanh not implemented for e2cnn")
    elif settings['models'][model_name]['acti'] == 'elu':
        if model_name == 'UNet_vanilla' or model_name == 'UNet_bcnn':
            acti = nn.ELU
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
        elif model_name == 'UNet_e2cnn':
            acti = e2_nn.ELU
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
    elif settings['models'][model_name]['acti'] == 'sigmoid':
        if model_name == 'UNet_bcnn':
            acti = nn.Sigmoid
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
        else:
            raise NotImplementedError("sigmoid not implemented for vanilla and e2cnn")
    elif settings['models'][model_name]['acti'] == 'custom':
        if model_name == 'UNet_bcnn':
            acti = paramActivation
            acti_kwargs = settings['models'][model_name]['acti_kwargs']
        else:
            raise NotImplementedError("custom not implemented for vanilla and e2cnn")
    else:
        raise ValueError("Unknown activation function: should be 'relu', 'tanh' or 'elu'")

    if model_name == 'UNet_vanilla':
        model = UNet_vanilla(settings['in_channels'], settings['n_classes'], 
                             lbda=settings['models'][model_name]['lbda'], acti=acti, 
                             acti_kwargs=acti_kwargs, bilinear=settings['models'][model_name]['bilinear'],
                             kernel_size=settings['models'][model_name]['kernel_size'])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, n_params
    elif model_name == 'UNet_bcnn':
        model = UNet_bcnn(settings['in_channels'], settings['n_classes'],
                          lbda=settings['models'][model_name]['lbda'], acti=acti, 
                          acti_kwargs=acti_kwargs, bilinear=settings['models'][model_name]['bilinear'],
                          kernel_size=settings['models'][model_name]['kernel_size'],
                          reflex_inv=settings['models'][model_name]['reflex_inv'],
                          scale_inv=settings['models'][model_name]['scale_inv'],
                          cutoff=settings['models'][model_name]['cutoff'],
                          TensorCorePad=settings['models'][model_name]['TensorCorePad'],
                          mode=settings['models'][model_name]['mode'],
                          bn=settings['models'][model_name]['bn'])
        
        # Loop through the different layers to count the number of parameters
        n_params = 0
        for layer in model.children():
            for sublayer in layer.children():
                for subsublayer in sublayer.children():
                    for layer in subsublayer.children():
                        if layer.__class__.__name__ == 'BesselConv2d':
                            n_params += layer.n_params
                        else:
                            n_params += sum(p.numel() for p in layer.parameters() if p.requires_grad)

        return model, n_params
    
    elif model_name == 'UNet_e2cnn':
        model = UNet_e2cnn(settings['in_channels'], settings['n_classes'],
                           lbda=settings['models'][model_name]['lbda'], acti=acti, 
                           acti_kwargs=acti_kwargs, bilinear=settings['models'][model_name]['bilinear'],
                           kernel_size=settings['models'][model_name]['kernel_size'],
                           mode=settings['models'][model_name]['mode'])
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model, n_params
