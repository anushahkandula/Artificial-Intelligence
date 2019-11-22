
import torch
from torch import ones
import numpy as np

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
       
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        self.netmodel = torch.nn.Sequential(torch.nn.Linear(in_size, 32), torch.nn.LeakyReLU(), torch.nn.Linear(32, 32), torch.nn.LeakyReLU(), torch.nn.Linear(32, out_size), torch.nn.LeakyReLU())
        self.optimizer = torch.optim.Adam(self.netmodel.parameters(), lr=self.lrate)


    def set_parameters(self, params):

        pass

    def get_parameters(self):

        return self.netmodel.parameters()


    def forward(self, x):

        x=(x-(x.mean(dim=1,keepdim=True)))/(x.std(dim=1,keepdim=True))
        return self.netmodel(x)


    def step(self, x, y):

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x), y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=1):

    # train_set=(train_set-torch.mean(train_set))/torch.std(train_set)
    loss_function = torch.nn.CrossEntropyLoss()
    net = NeuralNet(0.00001, loss_function, len(train_set[0]), 2)

    losses=np.zeros(n_iter)
    n = len(train_set[0])//batch_size

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_set,train_labels), batch_size=batch_size, shuffle=True)
    a=train_set.numpy()
    for iters in range(n_iter):
        loss=0
        for (inputs, labels) in train_loader:
            # inputs = train_set[(i*batch_size):((i+1)*batch_size)]
            # labels = train_labels[(i * batch_size):((i + 1) * batch_size)]
            loss+=net.step(inputs, labels)
        losses[iters]=loss


    yhats = np.zeros(len(dev_set))
    output=net.forward(dev_set)
    for i in range(len(dev_set)):
        yhats[i]=output[i][1] > output[i][0]

    return losses, yhats, net
