import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self, inp_size, out_size):
        super(Net, self).__init__()
        # defining or initialising the layers of the network
        self.model = nn.Linear(inp_size, out_size)

        self.model.cuda()
        self.optim = optim.Adam(self.model.parameters(), lr=1e-2)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, train_data, train_labels):
        train_data = torch.from_numpy(train_data).float()
        train_labels = torch.from_numpy(train_labels).long()

        train_dset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=4096, shuffle=True,
                                                   num_workers=4, pin_memory=True)

        self.model.train()
        optimizer = self.optim
        epochs = 100
        print_freq = 10

        for j in range(epochs):
            avg_loss = 0
            for i, (x, y) in enumerate(train_loader):
                # got a batch of data and labels
                x = x.cuda()
                y = y.cuda()
                out = self.model(x)
                # computed the neural network over the input x
                optimizer.zero_grad()
                # compute the loss
                loss = self.loss_fn(out, y)
                # backpropagating the loss
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss + loss.item()
            avg_loss = avg_loss / len(train_loader)
            print("the avg loss is ", avg_loss)

        return avg_loss

    def score(self, val_data, val_labels):
        val_data = torch.from_numpy(val_data).float()
        val_labels = torch.from_numpy(val_labels).long()

        val_dset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=4096, shuffle=False,
                                                 num_workers=4, pin_memory=True)

        self.model.eval()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(val_loader):
            x = x.cuda()
            out = self.model(x)
            pred = out.cpu().data.numpy().argmax(axis=1)
            correct += (pred == y.cpu().data.numpy()).sum()
            total += len(pred)
        print("the total is ", total)

        acc = float(correct) / float(total)
        return acc
