import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import sys

print(sys.argv[0])


class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()
        # define layers
        self.ly1 = nn.Linear(*input_dims, 128)
        self.ly2 = nn.Linear(128, 256)
        self.ly3 = nn.Linear(256, n_classes)

        # define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

        # use gpu
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("using: {}".format(self.device))
        self.to(self.device)

    def forward(self, data):
        layer1 = F.relu(self.ly1(data))
        layer2 = F.sigmoid(self.ly2(layer1))
        layer3 = self.ly3(layer2)
        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)

        preds = self.forward(data)

        cost = self.loss(preds, labels)

        cost.backward()
        self.optimizer.step()



















