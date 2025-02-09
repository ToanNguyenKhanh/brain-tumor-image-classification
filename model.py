"""
@author: <nktoan163@gmail.com>
"""
import torch
import torch.nn as nn
from torchsummary import summary
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.conv1 = self.block(inchannel=3, outchannel=8)
        self.conv2 = self.block(inchannel=8, outchannel=16)
        self.conv3 = self.block(inchannel=16, outchannel=32)
        self.conv4 = self.block(inchannel=32, outchannel=64)
        self.conv5 = self.block(inchannel=64, outchannel=128)
        self.conv6 = self.block(inchannel=128, outchannel=256)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=2304, out_features=num_classes),
            nn.LeakyReLU()
        )
    def block(self, inchannel, outchannel):
        return nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = CNN()
    model.train()
    sample_input = torch.rand(3, 3, 224, 224)
    result = model(sample_input)
    print(result.shape)
    model = model.to('cuda')
    summary(model, input_size=(3, 224, 224))
