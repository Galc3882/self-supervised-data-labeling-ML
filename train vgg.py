import os
import pickle
import progressbar
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


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
    model = VGG16(num_classes).to(device)

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
                   'model/vgg_{}.ckpt'.format(epoch+1))

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
        plt.savefig('loss_acc_vgg.png')
        plt.close()

    print('Finished Training')
