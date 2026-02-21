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

import torch

import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


class LFWPeople(torch.utils.data.Dataset):

    def __init__(self, mode):
        super().__init__()
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4, data_home="data")

        X = lfw_people.data
        y = lfw_people.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        if mode == "train":
            self.data = torch.from_numpy(X_train.reshape(-1, 50, 37))
            self.target = torch.from_numpy(y_train).long()
        else:
            self.data = torch.from_numpy(X_test.reshape(-1, 50, 37))
            self.target = torch.from_numpy(y_test).long()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx : idx + 1]
        target = self.target[idx]
        return img, target


class SimpleCNNModel(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.cnn1 = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.cnn2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.cnn3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.AdaptiveAvgPool2d((2, 2))

        self.fc1 = torch.nn.Linear(128 * 4, num_classes * 2)
        self.fc2 = torch.nn.Linear(num_classes * 2, num_classes)

    # (128, 1, h, w)
    def forward(self, data):

        o1 = self.cnn1(data)
        o2 = self.relu(self.bn1(o1))
        # ic(o1.shape)

        o3 = self.cnn2(o2)
        o4 = self.relu(self.bn2(o3))
        # ic(o3.shape)

        o5 = self.cnn3(o4)
        o6 = self.relu(self.bn3(o5))
        # ic(o6.shape)

        o7 = self.pool(o6)
        # ic(o7.shape)
        o7 = o7.reshape(o1.shape[0], -1)
        # ic(o7.shape)
        o7 = self.fc1(o7)
        result = self.fc2(o7)
        # ic(result.shape)

        return result


def main():
    device = torch.device("cuda:0")

    train_set = LFWPeople(mode="train")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)

    val_set = LFWPeople(mode="val")
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=True, num_workers=12, pin_memory=True)

    model = SimpleCNNModel(num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_function = torch.nn.CrossEntropyLoss()
    # ?accuracy
    total_epochs = 100
    max_epoch_acc = 0
    table = []

    for current_epoch in range(total_epochs):
        model.train()
        countLoss = 0
        totalLoss = 0
        # for data, target in tqdm(train_loader, desc=f"[{current_epoch:> 3}/{total_epochs}]", ncols=80):
        for data, target in train_loader:  # validation loader
            data, target = data.to(device), target.to(device)
            pred = model(data)
            # print(pred.shape, target.shape, pred.max(), target.max())
            loss = loss_function(pred, target)
            countLoss += 1
            totalLoss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avgLoss = totalLoss / countLoss

        # validate
        # 1. validation loader that is not equal to train loader
        model.eval()
        countBatch = 0
        totalAccuracy = 0

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
        print(table[-1])


if __name__ == "__main__":
    main()
