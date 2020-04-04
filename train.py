import math

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn


class GCN(nn.Module):
    def __init__(self, A_hat, num_feat, num_hidden, num_class):
        super(GCN, self).__init__()
        self.num_feat = num_feat
        self.num_hidden = num_hidden
        self.num_class = num_class
        self.A_hat = A_hat

        self.W_0 = nn.Parameter(torch.Tensor(num_feat, num_hidden))
        self.W_1 = nn.Parameter(torch.Tensor(num_hidden, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.W_0.size(1))
        self.W_0.data.uniform_(-stdv, stdv)
        stdv = 1 / math.sqrt(self.W_1.size(1))
        self.W_1.data.uniform_(-stdv, stdv)

    def forward(self, x, A_hat):
        H = torch.mm(torch.mm(A_hat, x), self.W_0)
        H = F.relu(H)
        H = torch.mm(torch.mm(A_hat, H), self.W_1)
        return F.log_softmax(H, dim=1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = nx.karate_club_graph()

    X = np.identity(G.number_of_nodes(), dtype=np.float)

    A = nx.adjacency_matrix(G).todense()
    L = nx.laplacian_matrix(G).todense()
    D = L + A

    A_tilde = np.array(A, dtype=np.float) + X
    D_temp = np.array(D, dtype=np.float)
    for i in range(G.number_of_nodes()):
        D_temp[i][i] = 1.0 / math.sqrt(D_temp[i][i])
    A_hat = np.matmul(np.matmul(D_temp, A_tilde), D_temp)

    y = []
    for node in G.nodes:
        if G.nodes[node]['club'] == 'Mr. Hi':
            y.append(0)
        elif G.nodes[node]['club'] == 'Officer':
            y.append(1)

    num_feat = len(G.nodes())
    num_hidden = 10
    num_class = 2

    model = GCN(A_hat, num_feat, num_hidden, num_class)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    A_hat_tensor = torch.Tensor(A_hat).to(device)
    X_tensor = torch.Tensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)

    loss_list = []
    acc_list = []
    for epoch in range(500):
        model.train()
        model.zero_grad()

        out = model(X_tensor, A_hat_tensor)
        loss = criterion(out, y_tensor)
        loss_list.append(loss.item())

        preds = torch.argmax(out, dim=1)
        acc = torch.mean(torch.eq(preds, y_tensor).type(torch.DoubleTensor))
        acc_list.append(acc.numpy())

        if (epoch + 1) % 10 == 0:
            print(f'{epoch + 1} Loss: {loss.item()} Acc: {acc.numpy()}')

        loss.backward()
        optimizer.step()

    sns.set()
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(loss_list)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Acc')
    ax2.plot(acc_list)
    plt.savefig('result.png')


if __name__ == '__main__':
    main()
