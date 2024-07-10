""" Full assembly of the parts to form the complete network """

from building_blocks import *


class UNet_vanilla(nn.Module):
    def __init__(self, n_channels, n_classes, lbda=1, bilinear=True, acti = nn.ReLU,
                 acti_kwargs = {'inplace': True}, kernel_size=3):
        super(UNet_vanilla, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv_vanilla(n_channels, 64//lbda, acti=acti, 
                                       acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.down1 = (Down_vanilla(64//lbda, 128//lbda, acti=acti, 
                                   acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.down2 = (Down_vanilla(128//lbda, 256//lbda, acti=acti, 
                                   acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.down3 = (Down_vanilla(256//lbda, 512//lbda, acti=acti, 
                                   acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        factor = 2 if bilinear else 1
        self.down4 = (Down_vanilla(512//lbda, 1024//lbda // factor, acti=acti, 
                                   acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.up1 = (Up_vanilla(1024//lbda, 512//lbda // factor, bilinear, acti=acti, 
                               acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.up2 = (Up_vanilla(512//lbda, 256//lbda // factor, bilinear, acti=acti, 
                               acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.up3 = (Up_vanilla(256//lbda, 128//lbda // factor, bilinear, acti=acti, 
                               acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.up4 = (Up_vanilla(128//lbda, 64//lbda, bilinear, acti=acti, 
                               acti_kwargs=acti_kwargs, kernel_size=kernel_size))
        self.outc = (OutConv(64//lbda, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    


class UNet_bcnn(nn.Module):
    def __init__(self, n_channels, n_classes, lbda=1, bilinear=True, bn=False, 
                 reflex_inv=False, scale_inv=False, cutoff='strong', kernel_size=5,
                 acti = nn.Tanh, acti_kwargs = {}, TensorCorePad=False, mode='pure'):
        super(UNet_bcnn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        if mode == 'pure':
            Up = Up_bcnn
            kwargs = {'reflex_inv': reflex_inv, 'scale_inv': scale_inv, 'cutoff': cutoff,
                      'TensorCorePad': TensorCorePad, 'kernel_size': kernel_size, 
                      'acti': acti, 'acti_kwargs': acti_kwargs, 'bn': bn}
        elif mode == 'mixed':
            Up = Up_vanilla
            kwargs = {'acti': acti, 'acti_kwargs': acti_kwargs, 'kernel_size': kernel_size}
        else:
            raise ValueError("Unknown mode: should be 'pure' or 'mixed'")

        self.inc = (DoubleConv_bcnn(n_channels, int(64/lbda), reflex_inv=reflex_inv, bn=bn,
                                    scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad,
                                    kernel_size=kernel_size, acti=acti, acti_kwargs=acti_kwargs))
        self.down1 = (Down_bcnn(int(64/lbda), int(128/lbda), reflex_inv=reflex_inv, bn=bn, 
                                scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad,
                                kernel_size=kernel_size, acti=acti, acti_kwargs=acti_kwargs))
        self.down2 = (Down_bcnn(int(128/lbda), int(256/lbda), reflex_inv=reflex_inv, bn=bn, 
                                scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad,
                                kernel_size=kernel_size, acti=acti, acti_kwargs=acti_kwargs))
        self.down3 = (Down_bcnn(int(256/lbda), int(512/lbda), reflex_inv=reflex_inv, bn=bn, 
                                scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad,
                                kernel_size=kernel_size, acti=acti, acti_kwargs=acti_kwargs))
        factor = 2 if bilinear else 1
        self.down4 = (Down_bcnn(int(512/lbda), int(1024/lbda) // factor, reflex_inv=reflex_inv, 
                                scale_inv=scale_inv, cutoff=cutoff, TensorCorePad=TensorCorePad,
                                kernel_size=kernel_size, acti=acti, acti_kwargs=acti_kwargs, bn=bn))
        self.up1 = (Up(int(1024/lbda), int(512/lbda) // factor, bilinear, **kwargs))
        self.up2 = (Up(int(512/lbda), int(256/lbda) // factor, bilinear, **kwargs))
        self.up3 = (Up(int(256/lbda), int(128/lbda) // factor, bilinear, **kwargs))
        self.up4 = (Up(int(128/lbda), int(64/lbda), bilinear, **kwargs))
        self.outc = (OutConv(int(64/lbda), n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    


class UNet_e2cnn(nn.Module):
    def __init__(self, n_channels, n_classes, lbda=1, bilinear=True, acti = e2_nn.ELU,
                 acti_kwargs = {'alpha': 0.1, 'inplace': True}, gspace = gspaces.Rot2dOnR2(N=4), 
                 kernel_size=5, mode='pure'):
        super(UNet_e2cnn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.mode = mode

        if mode == 'pure':
            Up = Up_e2cnn
            kwargs = {'acti': acti, 'acti_kwargs': acti_kwargs, 
                      'gspace': gspace, 'kernel_size': kernel_size}
        elif mode == 'mixed':
            Up = Up_vanilla
            kwargs = {'acti': nn.ReLU, 'acti_kwargs': {'inplace': True}, 'kernel_size': kernel_size}
        else:
            raise ValueError("Unknown mode: should be 'pure' or 'mixed'")

        self.inc = DoubleConv_e2cnn(n_channels, 64//lbda, first=True, acti=acti, 
                                    acti_kwargs=acti_kwargs, gspace=gspace, kernel_size=kernel_size)
        self.down1 = Down_e2cnn(64//lbda, 128//lbda, acti=acti, acti_kwargs=acti_kwargs, 
                                 gspace=gspace, kernel_size=kernel_size)
        self.down2 = Down_e2cnn(128//lbda, 256//lbda, acti=acti, acti_kwargs=acti_kwargs,
                                 gspace=gspace, kernel_size=kernel_size)
        self.down3 = Down_e2cnn(256//lbda, 512//lbda, acti=acti, acti_kwargs=acti_kwargs,
                                 gspace=gspace, kernel_size=kernel_size)
        factor = 2 if bilinear else 1
        self.down4 = Down_e2cnn(512//lbda, 1024//lbda // factor, acti=acti, acti_kwargs=acti_kwargs,
                                 gspace=gspace, kernel_size=kernel_size)
        self.up1 = Up(1024//lbda, 512//lbda // factor, bilinear, **kwargs)
        self.up2 = Up(512//lbda, 256//lbda // factor, bilinear, **kwargs)
        self.up3 = Up(256//lbda, 128//lbda // factor, bilinear, **kwargs)
        self.up4 = Up(128//lbda, 64//lbda, bilinear, **kwargs)

        in_type = e2_nn.FieldType(gspace, 64//lbda*[gspace.regular_repr])
        self.convert = ConvertToTensor(in_type)

        self.outc = (OutConv(64//lbda, n_classes))

    def forward(self, x):
        x1, x1_out_type = self.inc(x)
        x2, x2_out_type = self.down1(x1)
        x3, x3_out_type = self.down2(x2)
        x4, x4_out_type = self.down3(x3)
        x5, x5_out_type = self.down4(x4)
        if self.mode == 'pure':
            x = self.up1(x5, x4)
        else:
            x = self.up1(ConvertToTensor(x5_out_type)(x5), ConvertToTensor(x4_out_type)(x4))
        if self.mode == 'pure':
            x = self.up2(x, x3)
        else:
            x = self.up2(x, ConvertToTensor(x3_out_type)(x3))
        if self.mode == 'pure':
            x = self.up3(x, x2)
        else:
            x = self.up3(x, ConvertToTensor(x2_out_type)(x2))
        if self.mode == 'pure':
            x = self.up4(x, x1)
        else:
            x = self.up4(x, ConvertToTensor(x1_out_type)(x1))
        if self.mode == 'pure':
            x = self.convert(x)
        logits = self.outc(x)
        return logits
    


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    


if __name__ == '__main__':

    model = UNet_vanilla(3, 2, lbda=2)#.to('cuda')
    print(count_parameters(model))
    x = torch.randn((1, 3, 448, 448))#.to('cuda')
    print(model(x).shape)

    print("------------------------------------------------------------------")

    model = UNet_bcnn(3, 2, lbda=8, mode='mixed')#.to('cuda')
    print(count_parameters(model))
    x = torch.randn((1, 3, 448, 448))#.to('cuda')
    print(model(x).shape)

    print("------------------------------------------------------------------")

    gspace = gspaces.Rot2dOnR2(N=4)
    model = UNet_e2cnn(3, 2, lbda=8, gspace=gspace, mode='mixed')#.to('cuda')
    print(count_parameters(model))
    x = torch.randn((1, 3, 448, 448))#.to('cuda')
    print(model(x).shape)