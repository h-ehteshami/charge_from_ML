import torch 
import torch.nn as nn

class torch_mlp(torch.nn.Module):
    def __init__(self, n_featres, shape, n_outputs):
        super(torch_mlp, self).__init__()
        
        layers = []; shape.insert(0, n_featres)
        for i in range(len(shape)-1):
            layers.append( nn.Linear(shape[i], shape[i+1]) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(shape[-1], n_outputs) )

        self.layers = nn.Sequential( *layers )
    
    def forward(self, inpt):
        x = self.layers(inpt)
        return x

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        layers = args.layers
        size = args.kernal_size 
        input_layers = args.input_layers
        self.input_layers = input_layers
        residual = args.residual
        self.residual = residual

        #self.normalization = args.normalization
        activation_function = nn.ReLU(inplace = True)
        pooling_layer = nn.MaxPool3d(kernel_size=2)

        self.initial = nn.Sequential(
            # input is 3 x D x D x D
            nn.Conv3d(in_channels=input_layers,
                      out_channels=layers,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular",
                      bias=False),
            activation_function,

            nn.Conv3d(in_channels=layers,
                      out_channels=layers,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular",
                      bias=False),
            activation_function
        )

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=layers,
                      out_channels=1,
                      kernel_size=(1,1,1),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular",
                      bias=False)
        )

        # Encoder
        self.encoder1 = Encoder(in_channels=layers,
                                out_channels=2*layers,
                                kernal_size=size,
                                activation_function=activation_function,
                                pooling_layer=pooling_layer
                               )
        self.encoder2 = Encoder(in_channels=2*layers,
                                out_channels=4*layers,
                                kernal_size=size,
                                activation_function=activation_function,
                                pooling_layer=pooling_layer
                               )
        self.encoder3 = Encoder(in_channels=4*layers,
                                out_channels=8*layers,
                                kernal_size=size,
                                activation_function=activation_function,
                                pooling_layer=pooling_layer
                               )

        # Decoder
        self.decoder1 = Decoder(in_channels=8*layers,
                                out_channels=4*layers,
                                kernal_size=size,
                                activation_function=activation_function
                               )
        self.decoder2 = Decoder(in_channels=4*layers,
                                out_channels=2*layers,
                                kernal_size=size,
                                activation_function=activation_function
                               )
        self.decoder3 = Decoder(in_channels=2*layers,
                                out_channels=layers,
                                kernal_size=size,
                                activation_function=activation_function
                               )

    def forward(self, input):

        x = input[:,1,:,:,:] #* self.std + self.mean

        residual = self.residual
        input_layers = self.input_layers


        if input_layers == 2:
            pass
        elif input_layers == 1:
            input = input[:,1,:,:,:].unsqueeze(1)

        x0 = self.initial(input)

        # encoding layers
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # decoding layers
        out = self.decoder1(x3, x2)
        out = self.decoder2(out, x1)
        out = self.decoder3(out, x0)

        # final layers
        out = self.final(out)
        out = out.squeeze(dim=1)

        relu = nn.ReLU(inplace=True)
        if (residual):
            out = relu(out+x)
        else:
            out = relu(out)

        return out

        # normalization
        #if (self.normalization):
        #    charge = torch.sum(out,(3,2,1))
        #    scale = (charge0/charge).view(-1,1,1,1)
        #    return scale * out
        #else:
        #    return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, activation_function):
        super(Decoder, self).__init__()
        size = kernal_size
        # upsampling
        self.upsampling = nn.ConvTranspose3d(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=(2,2,2),
                                             stride=(2,2,2),
                                             padding=0,
                                             bias=False)

        # convolution
        self.convolution1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular", 
                      bias=False),
            activation_function
        )
        self.convolution2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular", 
                      bias=False),
            activation_function
        )

    def forward(self, x, x0):
        out = self.upsampling(x)
        out = torch.cat((out, x0), dim=1) # skip connection
        out = self.convolution1(out)
        out = self.convolution2(out)
        return out


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size, activation_function, pooling_layer):
        super(Encoder, self).__init__()

        size = kernal_size
        self.downsampling = nn.Sequential(
            pooling_layer,
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular", 
                      bias=False),
            activation_function,

            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode="circular", 
                      bias=False),
            activation_function
        )

    def forward(self, x):
        return self.downsampling(x)


