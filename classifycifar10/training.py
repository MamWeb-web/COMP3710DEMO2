"""
data: storage, read, load, process
model: architecture, forward

training
    - initialize model, data, optimizer, loss function
    - for batch, batchtarget in data:
        prediction = model(batch)
        loss = lossfunction(prediction, batchtarget)
        loss.backward()
        optimizer.step()

auxiliary:
    1. parameters -> data path, model layers, learnign rate
    2. validation
    3. store
    4. training speed up (AMP, accelerate)
    ....

"""

import pickle
import tarfile

import torch
from torch.amp import GradScaler

from tqdm import trange

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, mode, path="data/cifar-10-python.tar.gz"):
        super().__init__()

        X, y = [], []
        with tarfile.open(path, "r:gz") as tar:
            for member in tar.getmembers():
                if "data_batch" in member.name or "test_batch" in member.name:
                    f = tar.extractfile(member)
                    batch = pickle.load(f, encoding="bytes")
                    X.append(batch[b"data"])
                    y.append(batch[b"labels"])

        X = np.concatenate(X).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.concatenate(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if mode == "train":
            self.data = torch.from_numpy(X_train)
            self.target = torch.from_numpy(y_train).long()
        else:
            self.data = torch.from_numpy(X_test)
            self.target = torch.from_numpy(y_test).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]  # shape: (3, 32, 32)
        target = self.target[idx]
        return img, target


class ResConv(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cnn = torch.nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        o1 = self.cnn(data)
        o2 = self.bn(o1)
        return self.relu(o2) + data


class SimpleCNNModel(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            ResConv(128),
            # torch.nn.AvgPool2d(2, 2),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
            # ResConv(128),
        )

        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        self.down = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=16, stride=16, padding=0, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(128 * 4, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        features = self.body(data)
        small = (self.pool(features) + self.down(features)) / 2

        o12 = small.reshape(data.shape[0], -1)
        o12 = self.dropout(o12)

        o13 = self.relu(self.fc1(o12))
        logits = self.fc2(o13)
        return logits


def main():
    device = torch.device("cuda:0")
    scaler = GradScaler()
    total_epochs = 100

    train_set = CIFAR10(mode="train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)

    val_set = CIFAR10(mode="val")
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True, num_workers=12, pin_memory=True)

    model = SimpleCNNModel(num_classes=10).to(device)
    # model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epochs)
    loss_function = torch.nn.CrossEntropyLoss()
    # ?accuracy
    max_epoch_acc = 0
    table = []

    tbar = trange(total_epochs, ncols=80)
    # for current_epoch in range(total_epochs):
    for current_epoch in tbar:
        model.train()
        countLoss = 0
        totalLoss = 0
        for data, target in train_loader:  # validation loader

            data, target = data.half().to(device), target.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(data)
                loss = loss_function(pred, target)

            countLoss += 1
            totalLoss += loss.item()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()
        avgLoss = totalLoss / countLoss
        scheduler.step()

        # validate
        # 1. validation loader that is not equal to train loader
        model.eval()
        countBatch = 0
        totalAccuracy = 0

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            # with torch.no_grad():
            for data, target in val_loader:
                total = 0
                data = data.half().to(device)  # move data to device
                target = target.to(device)
                pred = model(data)
                # accuracy
                _, pred = torch.max(pred, dim=1)
                total += target.shape[0]
                equal = pred == target
                accuracy = sum(equal) / total * 100
                countBatch += 1
                # accuracy for each batch
                # count (# of batchs)
                totalAccuracy += accuracy
            avgAccuracy = totalAccuracy / countBatch

        # save avg accuracy to csv and update file
        # save
        table.append((current_epoch, float(avgLoss), float(avgAccuracy)))
        # torch.save(model.state_dict(), 'model_weights.pth')
        current_epoch_acc = avgAccuracy
        if current_epoch_acc > max_epoch_acc:
            max_epoch_acc = current_epoch_acc
            torch.save(model.state_dict(), "model_weights.pth")

        pd_table = pd.DataFrame(
            table,
            columns=["epoch", "train_ce", "val_acc"],
        )
        pd_table.to_csv("record.csv", index=False)
        # print(table[-1])
        tbar.set_description(str(round(table[-1][-1], 2)))


if __name__ == "__main__":
    main()
