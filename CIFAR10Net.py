from torch import nn
import nn.functional as F


class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.batch1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.batch2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)

        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool3 = nn.MaxPool2d(2, 2)

        self.batch3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 4 * 4, 128)

        self.fc2 = nn.Linear(128, 64)

        self.fc3 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = self.batch1(self.pool1(F.relu(self.conv2(x))))

        x = F.relu(self.conv3(x))

        x = self.batch2(self.pool2(F.relu(self.conv4(x))))

        x = F.relu(self.conv5(x))

        x = self.batch3(self.pool3(F.relu(self.conv6(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)
        return x
