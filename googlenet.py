import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
	def __init__(
		self,
		in_channels,
		ch1x1,
		ch3x3red,
		ch3x3,
		ch5x5red,
		ch5x5,
		pooling
		):
		super(Inception, self).__init__()

		# 1x1 conv branch
		self.branch1 = nn.Sequential(
			nn.Conv2d(in_channels, ch1x1, kernel_size=1),
			nn.BatchNorm2d(ch1x1),
			nn.ReLU(True)
			)

		# 1x1 conv + 3x3 conv branch
		self.branch2 = nn.Sequential(
			nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
			nn.BatchNorm2d(ch3x3red),
			nn.ReLU(True),
			nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
			nn.BatchNorm2d(ch3x3),
			nn.ReLU(True)
			)

		# 1x1 conv + 5x5 conv branch
		self.branch3 = nn.Sequential(
		 	nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
		 	nn.BatchNorm2d(ch5x5red),
		 	nn.ReLU(True),
		 	nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
		 	nn.BatchNorm2d(ch5x5),
		 	nn.ReLU(True)
		 	)

		# 3x3 pool + 1x1 conv branch
		self.branch4 = nn.Sequential(
		  	nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
		  	nn.Conv2d(in_channels, pooling, kernel_size=1),
	        nn.BatchNorm2d(pooling),
	        nn.ReLU(True)
		  	)

	def forward(self, x):
		branch1 = self.branch1(x)
		#print(branch1.shape)
		branch2 = self.branch2(x)
		#print(branch2.shape)
		branch3 = self.branch3(x)
		#print(branch3.shape)
		branch4 = self.branch4(x)
		#print(branch4.shape)
		return torch.cat([branch1, branch2, branch3, branch4], 1)

class GoogLeNet(nn.Module):
	def __init__(self, num_classes):
		super(GoogLeNet, self).__init__()

		#conv layers before inception
		self.pre_inception = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.MaxPool2d(3, stride=2, ceil_mode=True),
			nn.ReLU(True),
			nn.Conv2d(64, 64, kernel_size=1),
			nn.ReLU(True),
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.MaxPool2d(3, stride=2, ceil_mode=True),
			nn.ReLU(True)
			)

		self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
		self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
		self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
		self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
		self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
		self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
		self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
		self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
		self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
		self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.dropout = nn.Dropout(0.4)
		self.fc = nn.Linear(1024, num_classes)

	def forward(self, x):
		x = self.pre_inception(x)
		#print('done')
		x = self.inception3a(x)
		x = self.inception3b(x)
		x = self.maxpool(x)
		x = self.inception4a(x)
		x = self.inception4b(x)
		x = self.inception4c(x)
		x = self.inception4d(x)
		x = self.inception4e(x)
		x = self.maxpool(x)
		x = self.inception5a(x)
		x = self.inception5b(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


model = GoogLeNet(1000)
x = torch.randn(16,3,224,224)
y = model(x)
print(y.size())