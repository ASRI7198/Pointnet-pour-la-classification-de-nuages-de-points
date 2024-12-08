import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from os import listdir
from os import makedirs
from os.path import join
from os.path import exists
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize(x):
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], color="blue")
    plt.title("Point Set")

    # show plot
    plt.show()


class DatasetFromFolder(data.Dataset):
    def __init__(self, datadir):
        super(DatasetFromFolder, self).__init__()

        datadir1 = join(datadir, '00')
        filenames1 = [join(datadir1, x) for x in listdir(datadir1)]

        datadir2 = join(datadir, '01')
        filenames2 = [join(datadir2, x) for x in listdir(datadir2)]

        datadir3 = join(datadir, '02')
        filenames3 = [join(datadir3, x) for x in listdir(datadir3)]

        self.filenames = filenames1 + filenames2 + filenames3

    def __getitem__(self, index):
        name = self.filenames[index]
        input = torch.from_numpy(np.loadtxt(name).transpose(1, 0)).type(torch.FloatTensor)

        target = torch.zeros([1], dtype=torch.long)
        if name.find("/00/") >= 0:
            target[0] = 0  # cylinder
        elif name.find("/01/") >= 0:
            target[0] = 1  # rectangle
        elif name.find("/02/") >= 0:
            target[0] = 2  # torus
        else:
            print("bug")

        return input, target

    def __len__(self):
        return len(self.filenames)


# transform
class MyTNet(nn.Module):
    def __init__(self, dim=3):
        super(MyTNet, self).__init__()
        self.conv1 = nn.Conv1d(dim, 64, 1, 1, 1)
        self.conv2 = nn.Conv1d(64, 128, 1, 1, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1, 1, 1)

        self.BatchNorm1 = nn.BatchNorm1d(64)
        self.BatchNorm2 = nn.BatchNorm1d(128)
        self.BatchNorm3 = nn.BatchNorm1d(1024)

        self.RelU = nn.ReLU()

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)

        self.BatchNormL1 = nn.BatchNorm1d(512)
        self.BatchNormL2 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, dim * dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.RelU(x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.RelU(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.RelU(x)

        x = torch.max(x, dim=2, keepdim=True)[0]

        x = self.linear1(x)
        x = self.BatchNormL1(x)
        x = self.RelU(x)

        x = self.linear2(x)
        x = self.BatchNormL2(x)
        x = self.RelU(x)

        x = self.linear3(x)
        x = self.BatchNormL3(x)
        x = self.RelU(x)

        # x.size()[0] ==> batch size
        # adding to the identity matrix
        myidentity = torch.from_numpy(np.eye(self.dim, dtype=np.float32)).view(1, self.dim * self.dim).repeat(
            x.size()[0], 1)
        if x.is_cuda:
            myidentity = myidentity.cuda()
        x = x + myidentity
        x = x.view(-1, self.dim, self.dim)

        return x


# PointNet
class MyPointNet(nn.Module):
    def __init__(self, dim=3, dimfeat=64, num_class=3):
        super(MyPointNet, self).__init__()
        self.Tnet1 = MyTNet(dim)
        self.conv1 = nn.Conv1d(dim, dimfeat, 1, 1, 1)
        self.BatchNorm1 = nn.BatchNorm1d(64)

        self.Relu = nn.ReLU()

        self.Tnet2 = MyTNet(dimfeat)
        self.conv2 = nn.Conv1d(dimfeat, 128, 1, 1, 1)
        self.BatchNorm2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv3d(128, 1024, 1, 1, 1)
        self.BatchNorm3 = nn.BatchNorm1d(1024)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)

        self.BatchNormL1 = nn.BatchNorm1d(512)
        self.BatchNormL2 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, num_class)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x_Tnet1 = self.Tnet1(x)
        x = torch.bmm(x_Tnet1, x)

        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = self.Relu(x)

        x_Tnet2 = self.Tnet2(x)
        x = torch.bmm(x_Tnet2, x)

        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.Relu(x)

        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.Relu(x)

        x = torch.max(x, dim=2, keepdim=True)[0]

        x = self.linear1(x)
        x = self.BatchNormL1(x)
        x = self.Relu(x)

        x = self.linear2(x)
        x = self.BatchNormL2(x)
        x = self.Relu(x)

        x = self.linear3(x)
        x = self.log_softmax(x, dim=1)

        return x


myptnet = MyPointNet()

if not exists('mypointnet.pt'):
    myptnet.to(device)

    num_epochs = 100
    num_w = 4
    batch_s = 32

    # Loss and optimizer
    optimizer = optim.SGD(myptnet.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.NLLLoss()

    # loading data
    trainset = DatasetFromFolder("data/train")
    trainloader = DataLoader(trainset, num_workers=num_w, batch_size=batch_s, shuffle=True)
    losslog = []

    # train
    for epoch in range(0, num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = myptnet(inputs)
            loss = criterion(outputs, labels.squeeze(1))
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            epoch_loss += loss.item()

        # Log epoch loss
        avg_loss = epoch_loss / len(trainloader)
        losslog.append(avg_loss)
        print(f"{epoch} Epoch - training loss: {avg_loss:.4f}")


    # display the result
    plt.figure(figsize=(6, 4))
    plt.yscale('log')
    plt.plot(losslog, label='loss ({:.4f})'.format(losslog[-1]))
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    plt.close()

    torch.save(myptnet.state_dict(), 'mypointnet.pt')

else:
    # read the saved model
    myptnet.load_state_dict(torch.load('mypointnet.pt'))
    myptnet.eval()

testset = DatasetFromFolder("data/test")
testloader = DataLoader(testset, num_workers=4, batch_size=1, shuffle=False)

gtlabels = []
predlabels = []

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU if available

    with torch.no_grad():
        outputs = myptnet(inputs)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get predicted labels

    gtlabels.extend(labels.cpu().numpy())  # Append ground truth labels
    predlabels.extend(predicted.cpu().numpy())  # Append predicted labels


cm = confusion_matrix(gtlabels, predlabels)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()
