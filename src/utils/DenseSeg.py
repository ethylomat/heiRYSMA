import torch
import torch.nn as nn


class DenseNetSeg3D(nn.Module):
    # growth_rate in our case 16
    def __init__(self, device):
        super(DenseNetSeg3D, self).__init__()

        self.batch_norm = nn.BatchNorm3d(num_features=32)
        self.relu = nn.ReLU()

        # 64x64x64
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv3 = nn.Conv3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.conv_down = nn.Conv3d(32, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))

        self.growth_rate = 16
        self.reduction = 0.5

        self.dense_block_1 = DenseBlock(in_channels=32, n_layers=4, growth_rate=self.growth_rate, device=device)
        self.deconvolution1 = torch.nn.DataParallel(Deconvolution(in_channels=96, kernel_size=(4, 4, 4), stride=(2, 2, 2))).to(device)
        self.transition1 = torch.nn.DataParallel(TransitionLayer(in_channels=96, reduction=self.reduction)).to(device)

        self.dense_block_2 = DenseBlock(in_channels=48, n_layers=4, growth_rate=self.growth_rate, device=device)
        self.deconvolution2 = torch.nn.DataParallel(Deconvolution(in_channels=112, kernel_size=(6, 6, 6), stride=(4, 4, 4))).to(device)
        self.transition2 = torch.nn.DataParallel(TransitionLayer(in_channels=112, reduction=self.reduction)).to(device)

        self.dense_block_3 = DenseBlock(in_channels=56, n_layers=4, growth_rate=self.growth_rate, device=device)
        self.deconvolution3_128 = torch.nn.DataParallel(Deconvolution(in_channels=120, kernel_size=(10, 10, 14), stride=(8, 8, 8))).to(device)
        self.deconvolution3_64 = torch.nn.DataParallel(Deconvolution(in_channels=120, kernel_size=(10,10,10), stride=(8,8,8))).to(device)
        self.transition3 = torch.nn.DataParallel(TransitionLayer(in_channels=120, reduction=self.reduction)).to(device)

        self.dense_block_4 = DenseBlock(in_channels=60, n_layers=4, growth_rate=self.growth_rate, device=device)
        self.batch_norm2_3D = torch.nn.DataParallel(nn.BatchNorm3d(124)).to(device)
        self.deconvolution4_128 = torch.nn.DataParallel(Deconvolution(in_channels=124, kernel_size=(18, 18, 22), stride=(16, 16, 16))).to(device)
        self.deconvolution4_64 = torch.nn.DataParallel(Deconvolution(in_channels=124, kernel_size=(18, 18, 18), stride=(16, 16, 16))).to(device)
        self.conv4 = torch.nn.DataParallel(nn.Conv3d(96, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1))).to(device)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, target_resolution):
        x = torch.unsqueeze(x, 1)

        # 3x initial Convolution, after this layers output should be 64x64x64 and 32 channels
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        x1 = self.conv3(x)
        x = self.batch_norm(x1)
        x = self.relu(x)

        # first convolution for Downsampling
        x = self.conv_down(x)

        # DenseBlock 1
        x = self.dense_block_1(x)

        # Deconvolution 1
        x2 = self.deconvolution1(x)

        # Transition 1 -> includes Downsampling
        x = self.transition1(x)

        # DenseBlock 2
        x = self.dense_block_2(x)

        # Deconvolution 2
        x3 = self.deconvolution2(x)

        # Transition 2 -> includes Downsampling
        x = self.transition2(x)

        # DenseBlock 3
        x = self.dense_block_3(x)

        # Deconvolution 3
        if target_resolution == (128,128,100):
            x4 = self.deconvolution3_128(x)
        else:
            # 64x64x64 is default!
            x4 = self.deconvolution3_64(x)

        # Transition 3 -> includes Downsampling
        x = self.transition3(x)

        # DenseBlock 4
        x = self.dense_block_4(x)

        x = self.batch_norm2_3D(x)
        x = self.relu(x)

        # Deconvolution 4
        if target_resolution == (128,128,100):
            x5 = self.deconvolution4_128(x)
        else:
            #64x64x64 is default!
            x5 = self.deconvolution4_64(x)

        # concatenation of upscaled
        x = torch.cat([x5, x4, x3, x2, x1], dim=1)

        x = self.conv4(x)

        x = self.sigmoid(x)
        return x


class DenseBlock(nn.Module):
    '''
    in_channel in this case = growth rate?!
    '''

    def __init__(self, in_channels, n_layers, growth_rate, device):
        super(DenseBlock, self).__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.dense_layers = []

        for i in range(self.n_layers):
            model = DenseLayer(int(i * self.growth_rate + self.in_channels), self.growth_rate)
            model = model.double()
            dense_layer = torch.nn.DataParallel(model)
            dense_layer.to(device)
            self.dense_layers.append(dense_layer)

    def forward(self, x):
        for i in range(self.n_layers):
            # increases with each layer e.g. start with 64 -> 96 -> 128..

            # in_channels are out_channels from the last layer
            y = self.dense_layers[i](x)
            x = torch.cat([x, y], dim=1)
        return x


class DenseLayer(nn.Module):
    '''
    All Convolution Layers in this networks belongs to this BN-ReLU-Conv implementation
    '''

    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.batch_norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels, growth_rate, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, x):
        # first conv(1x1x1)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv1(x)

        # second conv(3x3x3)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)

        # after each Conv (3x3x3) dropout layer
        x = self.drop_out(x)

        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, reduction):
        super(TransitionLayer, self).__init__()
        self.batch_norm = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels, int(in_channels * reduction), kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        # first conv(1x1x1)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv1(x)

        # second downscaling conv(3x3x3)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x


class Deconvolution(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(Deconvolution, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels, 16, kernel_size=kernel_size, stride=stride,
                                          padding=(1, 1, 1))
        self.deconv1 = self.deconv1

    def forward(self, x):
        x = self.deconv1(x)

        return x
