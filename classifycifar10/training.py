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
import torchvision.transforms as T

from tqdm import trange

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, mode, transform=None, path="data/cifar-10-python.tar.gz"):
        super().__init__()

        self.transform = transform
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
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def main():
    device = torch.device("cuda:0")
    scaler = GradScaler()
    total_epochs = 150

    # Standard CIFAR-10 augmentation (already tensors, so no ToTensor needed)
    train_transform = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # Normalize to standard CIFAR-10 channel stats
            T.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    val_transform = T.Compose(
        [
            T.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    train_set = CIFAR10(mode="train", transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True, num_workers=12, pin_memory=True)

    val_set = CIFAR10(mode="val", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=True, num_workers=12, pin_memory=True)

    from resnet18 import ResNet

    model = ResNet([4, 6, 8, 6], num_classes=10)
    model = torch.compile(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-5)

    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # ?accuracy
    max_epoch_acc = 0
    table = []

    tbar = trange(total_epochs, ncols=80)
    for current_epoch in tbar:
        model.train()
        countLoss = 0
        totalLoss = 0
        for data, target in train_loader:

            data, target = data.to(device), target.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(data)
                loss = loss_function(pred, target)

            countLoss += 1
            totalLoss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
        avgLoss = totalLoss / countLoss

        if current_epoch < 100:
            scheduler.step()

        # validate
        # 1. validation loader that is not equal to train loader
        model.eval()
        countBatch = 0
        totalAccuracy = 0

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for data, target in val_loader:
                total = 0
                data = data.to(device)  # move data to device
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
