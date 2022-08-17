import os
import pickle
import progressbar
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels *
                               self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.expansion = 4
        layers = [3, 8, 36, 3]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNetLayers
        self.layer1 = self.make_layers(
            layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layers(
            layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layers(
            layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layers(
            layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layers(self, num_residual_blocks, intermediate_channels, stride):
        layers = []

        identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, intermediate_channels*self.expansion, kernel_size=1, stride=stride),
                                            nn.BatchNorm2d(intermediate_channels*self.expansion))
        layers.append(
            Block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * self.expansion  # 256
        for i in range(num_residual_blocks - 1):
            # 256 -> 64, 64*4 (256) again
            layers.append(Block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)


class getDataset(Dataset):

    def __init__(self, X_Train, Y_Train, imgTransform, labelTtransform):
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.imgTransform = imgTransform
        self.labelTtransform = labelTtransform

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        x = self.imgTransform(x)
        y = self.labelTtransform(y)

        return x, y


def labelTransform(label):
    """
    change labels from string to int ('go' -> 0, 'stop' -> 1, 'warning' -> 2)
    and then to tensor
    """
    return torch.tensor(0 if label == 'go' else 1 if label == 'stop' else 2)


if __name__ == '__main__':
    # get pickled train data
    with open('dataset/LISA_train_dataset.pickle', 'rb') as f:
        trainImages, trainLabels = pickle.load(f)
        f.close()

    # get pickled test data
    with open('dataset/LISA_test_dataset.pickle', 'rb') as f:
        testImages, testLabels = pickle.load(f)
        f.close()

    # get pickled mit data
    with open('dataset/MIT_test_dataset.pickle', 'rb') as f:
        mitImages, mitLabels = pickle.load(f)
        f.close()

    # # get pickled self supervised labels
    # with open('dataset/selfSupravisedTrainLabels.pickle', 'rb') as f:
    #     selfSupravisedTrainLables = pickle.load(f)
    #     f.close()
    # with open('dataset/selfSupravisedTestLabels.pickle', 'rb') as f:
    #     selfSupravisedTestLables = pickle.load(f)
    #     f.close()

    # initialize hyperparameters
    num_classes = 3
    batch_size = 8
    num_epochs = 20
    learning_rate = 0.0001
    shuffle = True
    weight_decay = 0.0001
    device = 'cuda'
    num_workers = 0

    # image transformation
    imgTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ])

    # load data
    trainDataset = getDataset(trainImages, trainLabels,
                              imgTransform, labelTransform)
    testDataset = getDataset(testImages, testLabels,
                             imgTransform, labelTransform)
    mitDataset = getDataset(mitImages, mitLabels, imgTransform, labelTransform)

    # load data loader
    trainLoader = DataLoader(trainDataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    testLoader = DataLoader(testDataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    mitLoader = DataLoader(mitDataset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    # create model
    model = ResNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # decrease learning rate if loss does not decrease
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True)

    # set cuda to optimize
    torch.backends.cudnn.benchmark = True

    # save statistics for analysis
    train_loss = []
    test_loss = []
    test_acc = []
    mit_loss = []
    mit_acc = []

    # Train the model
    for epoch in range(num_epochs):
        # initialize progress bar
        pbar = progressbar.ProgressBar(maxval=len(trainLoader), widgets=["Training model: ", progressbar.Percentage(
        ), " ", progressbar.Bar(), " ", progressbar.ETA()]).start()

        totalLoss = 0
        for i, (images, labels) in enumerate(trainLoader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            totalLoss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # progress bar
            pbar.update(i)

        # decrease learning rate if loss does not decrease
        scheduler.step(totalLoss/len(trainLoader))
        pbar.finish()

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, totalLoss/len(trainLoader)))

        # save statistics for analysis
        train_loss.append(totalLoss/len(trainLoader))

        # Validation
        totalLoss = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in testLoader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                totalLoss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs, predicted

            print('Accuracy of the network on the {} validation images: {} %'.format(
                len(testLoader)*batch_size, 100 * correct / total))
            print('Loss of the network on the {} validation images: {}'.format(
                len(testLoader)*batch_size, totalLoss/len(testLoader)))

            # save statistics for analysis
            test_loss.append(totalLoss/len(testLoader))
            test_acc.append(100 * correct / total)

            del correct, total

        # Test on MIT dataset
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in mitLoader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                totalLoss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs, predicted

            print('Accuracy of the network on the {} MIT images: {} %'.format(
                len(mitLoader)*batch_size, 100 * correct / total))
            print('Loss of the network on the {} MIT images: {}'.format(
                len(mitLoader)*batch_size, totalLoss/len(mitLoader)))

            # save statistics for analysis
            mit_loss.append(totalLoss/len(mitLoader))
            mit_acc.append(100 * correct / total)

            del correct, total

        # Save the model checkpoint
        # if folder does not exist, create it
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(model.state_dict(),
                   'model/resnet_{}.ckpt'.format(epoch+1))

        # plot loss and accuracy
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='train loss')
        plt.plot(test_loss, label='test loss')
        plt.plot(mit_loss, label='MIT loss')
        plt.legend()
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.subplot(1, 2, 2)
        plt.plot(test_acc, label='test acc')
        plt.plot(mit_acc, label='MIT acc')
        plt.legend()
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('loss_acc_resnet.png')
        plt.close()

    print('Finished Training')
