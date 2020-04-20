import torch.nn as nn
import torch.nn.functional as F

# Model A
drop_perc_in = 0.2
drop_perc_else = 0.5


class Model_A(nn.Module):
    def __init__(self):
        super(Model_A, self).__init__()
        self.drop_in = nn.Dropout2d(p=drop_perc_in)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5,
                               padding=1)
        self.drop = nn.Dropout2d(p=drop_perc_else)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1,
                               padding=1)
        self.avg = nn.AvgPool2d(kernel_size=6)  # What does averaging over 6x6 spatial dimensions mean?
        self.softmax = nn.Softmax(dim=1)

        # why no batch normalization???

    def forward(self, x):
        x = self.drop_in(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop(x)
        # print('size after conv1 and pooling', x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print('size after conv2 and pooling', x.size())
        x = F.relu(self.conv3(x))
        # print('size after conv3', x.size())
        x = F.relu(self.conv4(x))
        # print('size after conv4', x.size())
        x = F.relu(self.conv5(x))  # size after this layer [64, 10, 9, 9]
        # print('size after conv5', x.size())
        #x = self.avg(x)  # size after this layer [64, 10, 1, 1]
        x = F.adaptive_avg_pool2d(input=x, output_size=1)
        # print('size after averaging', x.size())
        x = x.squeeze()
        # print('size after squeezing', x.size())
        x = self.softmax(x)
        # print('size after softmax', x.size())
        return x


class Model_Strided_A(nn.Module):
    def __init__(self):
        super(Model_Strided_A, self).__init__()
        self.drop_in = nn.Dropout2d(p=drop_perc_in)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5,
                               padding=1, stride=2)
        self.drop = nn.Dropout2d(p=drop_perc_else)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5,
                               padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1,
                               padding=1)
        self.avg = nn.AvgPool2d(kernel_size=6)  # What does averaging over 6x6 spatial dimensions mean?
        self.softmax = nn.Softmax(dim=1)

        # why no batch normalization???

    def forward(self, x):
        x = self.drop_in(x)
        x = F.relu(self.conv1(x))
        x = self.drop(x)
        # print('size after conv1 and pooling', x.size())
        x = F.relu(self.conv2(x))
        # print('size after conv2 and pooling', x.size())
        x = F.relu(self.conv3(x))
        # print('size after conv3', x.size())
        x = F.relu(self.conv4(x))
        # print('size after conv4', x.size())
        x = F.relu(self.conv5(x))
        # print('size after conv5', x.size())
        x = F.adaptive_avg_pool2d(input=x, output_size=1)
        # print('size after averaging', x.size())
        x = x.squeeze()
        # print('size after squeezing', x.size())
        x = self.softmax(x)
        # print('size after softmax', x.size())
        return x


class Model_ConvPool_A(nn.Module):
    def __init__(self):
        super(Model_ConvPool_A, self).__init__()
        self.drop_in = nn.Dropout2d(p=drop_perc_in)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5,
                               padding=1)
        self.conv1a = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5,
                                padding=1)
        self.drop = nn.Dropout2d(p=drop_perc_else)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5,
                               padding=1)
        self.conv2a = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5,
                                padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1,
                               padding=1)
        self.avg = nn.AvgPool2d(kernel_size=6)  # What does averaging over 6x6 spatial dimensions mean?
        self.softmax = nn.Softmax(dim=1)

        # why no batch normalization???

    def forward(self, x):
        x = self.drop_in(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1a(x))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2a(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        #x = self.avg(x)  # size after this layer [64, 10, 1, 1]
        x = F.adaptive_avg_pool2d(input=x, output_size=1)
        x = x.squeeze()
        x = self.softmax(x)
        return x



class Model_All_A(nn.Module):
    def __init__(self):
        super(Model_All_A, self).__init__()
        self.drop_in = nn.Dropout2d(p=drop_perc_in)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5,
                               padding=1)
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5,
                                padding=1)  # what size should this be?
        self.drop = nn.Dropout2d(p=drop_perc_else)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=5,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1,
                               padding=1)
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1,
                               padding=1)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.avg = nn.AvgPool2d(kernel_size=6)  # What does averaging over 6x6 spatial dimensions mean?
        self.softmax = nn.Softmax(dim=1)

        # why no batch normalization???

    def forward(self, x):
        x = self.drop_in(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        #x = self.avg(x)  # size after this layer [64, 10, 1, 1]
        x = F.adaptive_avg_pool2d(input=x, output_size=1)
        # print('size after averaging', x.size())
        x = x.squeeze()
        # print('size after squeezing', x.size())
        x = self.softmax(x)
        # print('size after softmax', x.size())
        return x
