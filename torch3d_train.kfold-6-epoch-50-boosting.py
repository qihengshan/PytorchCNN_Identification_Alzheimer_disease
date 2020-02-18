from logger import FileLogger as Log
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

import numpy as np
import h5py

import torch.nn.functional as F
from datetime import datetime

start = datetime.now()


# define my class to read data
class MyDataset(Dataset):
    def __init__(self, transform=None, img_path=None, label_path=None):
        self.transform = transform
        self.img_path = img_path
        self.label_path = label_path
        self.labels = pd.read_csv(self.label_path)
        self.images_file = h5py.File(self.img_path)
        self.images = self.images_file['data']

    def __getitem__(self, index):
        # 3d convolution
        # so don't reshape
        return self.images[index], self.labels['label'][index]

    def __len__(self):
        return len(self.images)


class diseaseNet(nn.Module):

    def __init__(self, f=8):
        super(diseaseNet, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=8 * f, kernel_size=2, stride=1, padding=0))                   # (94, 78)
        self.conv.add_module('conv2', nn.InstanceNorm3d(num_features=8 * f))
        self.conv.add_module('conv3', nn.ReLU(inplace=True))
        self.conv.add_module('conv4', nn.MaxPool3d(kernel_size=4, stride=2))   # (46, 38)

        self.conv.add_module('conv5', nn.Conv3d(in_channels=8 * f, out_channels=32 * f, kernel_size=2, stride=1, dilation=1))  # (46, 38)
        self.conv.add_module('conv6', nn.InstanceNorm3d(num_features=32 * f))
        self.conv.add_module('conv7', nn.ReLU(inplace=True))
        self.conv.add_module('conv8', nn.MaxPool3d(kernel_size=4, stride=2))  # (22, 18)

        self.conv.add_module('conv9', nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=0, dilation=1))  # (22, 18)
        self.conv.add_module('conv10', nn.InstanceNorm3d(num_features=64 * f))
        self.conv.add_module('conv11', nn.ReLU(inplace=True))
        self.conv.add_module('conv12', nn.MaxPool3d(kernel_size=3, stride=2))  # (10, 8)

        self.conv.add_module('conv13', nn.Conv3d(in_channels=64 * f, out_channels=128 * f, kernel_size=2, stride=1, padding=0, dilation=1))  # (10, 8)
        self.conv.add_module('conv14', nn.InstanceNorm3d(num_features=128 * f))
        self.conv.add_module('conv15', nn.ReLU(inplace=True))
        self.conv.add_module('conv16', nn.MaxPool3d(kernel_size=3, stride=2)) # (4, 3)

        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(128 * f *  4 * 3, 1024))
        self.fc.add_module('dp1', nn.Dropout(0.3))
        # self.fc.add_module('fc2', nn.Linear(1024, 1024))
        # self.fc.add_module('dp2', nn.Dropout(0.3))
        self.fc.add_module('fc3', nn.Linear(1024, 256))
        self.fc.add_module('dp3', nn.Dropout(0.2))
        self.fc.add_module('fc4', nn.Linear(256, 3))


    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z.view(x.shape[0], -1))
        return z


# train_data = MyDataset(img_path='train_pre_data.h5', label_path='train_pre_label.csv')
# #train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=False)
# #print("Train data load success")

# # split train data set to (trian set , test set)
# train_set, test_set = torch.utils.data.random_split(train_data, [220, 80])

# # load data
# train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=False)
# print("Train data load success")
# test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=False)
# print("Test data load success")

train_target = pd.read_csv('train_pre_label.csv')
images_file = h5py.File('train_pre_data.h5', 'r')
train_features = images_file['data']


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_target = label_encoder.fit_transform(train_target['label'])

# Implement K-fold validation to improve results
n_splits = 5 # Number of K-fold Splits
splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(train_features, train_target))


## Hyperparameter
epochs = 50
batch_size = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = diseaseNet(f=10).cuda()
# print(model)
# print(type(model))

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       # transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       # transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# for each model: Compare accuracy Save the model with the highest accuracy
each_model_max_accuracy = []
# # Start K-fold validation
for k, (train_idx, valid_idx) in enumerate(splits):
    # model = diseaseNet(f=8).cuda()
    # print(model)
    x_train = np.array(train_features)
    y_train = np.array(train_target)

    x_train_fold = torch.tensor(x_train[train_idx.astype(int)], dtype=torch.float).to(device)
    y_train_fold = torch.tensor(y_train[train_idx.astype(int), np.newaxis], dtype=torch.float32).to(device)

    x_val_fold = torch.tensor(x_train[valid_idx.astype(int)], dtype=torch.float).to(device)
    y_val_fold = torch.tensor(y_train[valid_idx.astype(int), np.newaxis], dtype=torch.float32).to(device)

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    # loss func
    criterion = nn.CrossEntropyLoss()

    # model learn percentage
    mlps = [diseaseNet(f=8).cuda() for i in range(5)]
    optimizer = torch.optim.Adam([{"params": mlp.parameters()} for mlp in mlps],lr=0.0005, weight_decay=1e-5)
    steps = 0
    running_loss = [0 for num in range(len(mlps))]
    train_loss = []

    print(f"Fold {k + 1}")
    for epoch in range(epochs):
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  # N个Model清除梯度
            for j, model in enumerate(mlps):
                y_pred = model.forward(x_batch)
                loss = criterion(y_pred, y_batch.squeeze().long())
                loss.backward()
                running_loss[j] += loss.item()
            optimizer.step() #
        print(f"Epoch {epoch + 1} / {epochs}..")
        Log.info(f"Epoch {epoch + 1} / {epochs}..")
        for i in range(len(mlps)):
            train_loss.append(running_loss[i] / float(len(train)))
            print(f"{i} - Train loss: {train_loss[i]:.7f}..")
            Log.info(f"{i} - Train loss: {train_loss[i]:.7f}..")

        model.eval()

        pre = []
        vote_correct=0
        mlps_correct=[0 for i in range(len(mlps))]

        test_loss = [0 for i in range(len(mlps))]

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                for j, model in enumerate(mlps):
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch.squeeze().long())
                    test_loss[j] += loss.item()
                    pred_y_gpu = torch.max(y_pred, 1)[1].data
                    mlps_correct[j] += sum(pred_y_gpu == y_batch.squeeze())
                    pre.append(pred_y_gpu.cpu().numpy())

                arr = np.array(pre)
                pre.clear()
                # 注意split后，每一片要能被batch_size整除
                result = [Counter(arr[:, i]).most_common(1)[0][0] for i in range(batch_size)]
                vote_correct += (result == y_batch.squeeze().cpu().numpy()).sum()
            print("epoch:" + str(epoch) + "总的正确率"+str(float(vote_correct)/float(len(valid))))
            print(f"epoch: {epoch} - 总的正确率{vote_correct}/{len(valid)} - {float(vote_correct)/float(len(valid))}")
            Log.info("epoch: {} - 总的正确率:{}/{} --- {}".format(epoch, vote_correct, len(valid), float(vote_correct)/float(len(valid))))
            for i in range(len(mlps)):
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"{i} - Train loss: {train_loss[i]:.8f}.. "
                      f"{i} - Test loss: {test_loss[i]/float(len(valid)):.8f}.. "
                      f"{i} - Test accuracy({float(mlps_correct[i])}/{float(len(valid))}: {float(mlps_correct[i])/float(len(valid)):.8f}")
                Log.info(f"Epoch {epoch+1}/{epochs}.. {i} - Train loss: {train_loss[i]:.8f}.. Test loss: {test_loss[i]/float(len(valid)):.8f}.. Test accuracy({mlps_correct[i]}/{float(len(valid))}): {float(mlps_correct[i])/float(len(valid)):.8f}")
            for idx, coreect in enumerate( mlps_correct):
                print("网络"+str(idx)+"的正确率为："+str(float(coreect)/float(len(valid))))
                Log.info("网络 - {} - 正确率为：{}/{} --- {}".format(idx, coreect, len(valid), float(coreect)/float(len(valid))))

            if epoch > 20:
                if len(each_model_max_accuracy) < 1:
                    each_model_max_accuracy.extend(mlps_correct)
                    for i in range(len(mlps)):
                        torch.save(mlps[i], "boosting-model-{}-accuracy-{:.3f}.pth".format(i, float(mlps_correct[i])/float(len(valid))))
                else:
                    for i, acc in enumerate(each_model_max_accuracy):
                        if mlps_correct[i] > each_model_max_accuracy[i]:
                            each_model_max_accuracy[i] = mlps_correct[i]
                            torch.save(mlps[i], "boosting-model-{}-accuracy-{:.3f}.pth".format(i, float(mlps_correct[i])/float(len(valid))))
        # model.train()
# torch.save(model, "firstnetmodel-boosting-with-testset-kfold.pth")

stop = datetime.now()
print("Running time: ", stop-start)
