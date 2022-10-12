import copy
import numpy as np
import torch_geometric.nn as gnn
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.feature_selection import SelectKBest, f_classif


class GraphConvNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, alpha):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.input = nn.Linear(self.input_size, self.hidden_size[0], bias=False)
        self.alpha = alpha
        self.hiddens = nn.ModuleList([gnn.SAGEConv(self.hidden_size[h], self.hidden_size[h + 1]) for h in range(len(self.hidden_size) - 1)])
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        edge_index = self.create_edge_index(x)
        x = self.input(x)
        x = self.relu(x)
        for hidden in self.hiddens:
            x = hidden(x, edge_index)
            x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

    def create_edge_index(self, x):
        similarity_matrix = torch.abs(F.cosine_similarity(x[..., None, :, :], x[..., :, None, :], dim=-1))
        similarity = torch.sort(similarity_matrix.view(-1))[0]
        eps = torch.quantile(similarity, self.alpha, interpolation='nearest')
        adj_matrix = similarity_matrix >= eps
        row, col = torch.where(adj_matrix)
        edge_index = torch.cat((row.reshape(1, -1), col.reshape(1, -1)), dim=0)
        return edge_index


class GRACES:
    def __init__(self, n_features, hidden_size=None, q=2, n_dropouts=10, dropout_prob=0.5, batch_size=16,
                 learning_rate=0.001, epochs=50, alpha=0.95, sigma=0, f_correct=0):
        self.n_features = n_features
        self.q = q
        if hidden_size is None:
            self.hidden_size = [64, 32]
        else:
            self.hidden_size = hidden_size
        self.n_dropouts = n_dropouts
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.sigma = sigma
        self.f_correct = f_correct
        self.S = None
        self.new = None
        self.model = None
        self.last_model = None
        self.loss_fn = None
        self.f_scores = None

    @staticmethod
    def bias(x):
        if not all(x[:, 0] == 1):
            x = torch.cat((torch.ones(x.shape[0], 1), x.float()), dim=1)
        return x

    def f_test(self, x, y):
        slc = SelectKBest(f_classif, k=x.shape[1])
        slc.fit(x, y)
        return getattr(slc, 'scores_')

    def xavier_initialization(self):
        if self.last_model is not None:
            weight = torch.zeros(self.hidden_size[0], len(self.S))
            nn.init.xavier_normal_(weight, gain=nn.init.calculate_gain('relu'))
            old_s = self.S.copy()
            if self.new in old_s:
                old_s.remove(self.new)
            for i in self.S:
                if i != self.new:
                    weight[:, self.S.index(i)] = self.last_model.input.weight.data[:, old_s.index(i)]
            self.model.input.weight.data = weight
            for h in range(len(self.hidden_size) - 1):
                self.model.hiddens[h].lin_l.weight.data = self.last_model.hiddens[h].lin_l.weight.data
                self.model.hiddens[h].lin_r.weight.data = self.last_model.hiddens[h].lin_r.weight.data
            self.model.output.weight.data = self.last_model.output.weight.data

    def train(self, x, y):
        input_size = len(self.S)
        output_size = len(torch.unique(y))
        self.model = GraphConvNet(input_size, output_size, self.hidden_size, self.alpha)
        self.xavier_initialization()
        x = x[:, self.S]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        train_set = []
        for i in range(x.shape[0]):
            train_set.append([x[i, :], y[i]])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        for e in range(self.epochs):
            for data, label in train_loader:
                input_0 = data.view(data.shape[0], -1)
                optimizer.zero_grad()
                output = self.model(input_0.float())
                loss = self.loss_fn(output, label)
                loss.backward()
                optimizer.step()
        self.last_model = copy.deepcopy(self.model)

    def dropout(self):
        model_dp = copy.deepcopy(self.model)
        for h in range(len(self.hidden_size) - 1):
            h_size = self.hidden_size[h]
            dropout_index = np.random.choice(range(h_size), int(h_size * self.dropout_prob), replace=False)
            model_dp.hiddens[h].lin_l.weight.data[:, dropout_index] = torch.zeros(model_dp.hiddens[h].lin_l.weight[:, dropout_index].shape)
            model_dp.hiddens[h].lin_r.weight.data[:, dropout_index] = torch.zeros(model_dp.hiddens[h].lin_r.weight[:, dropout_index].shape)
        dropout_index = np.random.choice(range(self.hidden_size[-1]), int(self.hidden_size[-1] * self.dropout_prob), replace=False)
        model_dp.output.weight.data[:, dropout_index] = torch.zeros(model_dp.output.weight[:, dropout_index].shape)
        return model_dp

    def gradient(self, x, y, model):
        model_gr = GraphConvNet(x.shape[1], len(torch.unique(y)), self.hidden_size, self.alpha)
        temp = torch.zeros(model_gr.input.weight.shape)
        temp[:, self.S] = model.input.weight
        model_gr.input.weight.data = temp
        for h in range(len(self.hidden_size) - 1):
            model_gr.hiddens[h].lin_l.weight.data = model.hiddens[h].lin_l.weight + self.sigma * torch.randn(model.hiddens[h].lin_l.weight.shape)
            model_gr.hiddens[h].lin_r.weight.data = model.hiddens[h].lin_r.weight + self.sigma * torch.randn(model.hiddens[h].lin_r.weight.shape)
        model_gr.output.weight.data = model.output.weight
        output_gr = model_gr(x.float())
        loss_gr = self.loss_fn(output_gr, y)
        loss_gr.backward()
        input_gradient = model_gr.input.weight.grad
        return input_gradient

    def average(self, x, y, n_average):
        grad_cache = None
        for num in range(n_average):
            model = self.dropout()
            input_grad = self.gradient(x, y, model)
            if grad_cache is None:
                grad_cache = input_grad
            else:
                grad_cache += input_grad
        return grad_cache / n_average

    def find(self, input_gradient):
        gradient_norm = input_gradient.norm(p=self.q, dim=0)
        gradient_norm = gradient_norm / gradient_norm.norm(p=2)
        gradient_norm[1:] = (1 - self.f_correct) * gradient_norm[1:] + self.f_correct * self.f_scores
        gradient_norm[self.S] = 0
        max_index = torch.argmax(gradient_norm)
        return max_index.item()
    
    def select(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        self.f_scores = torch.tensor(self.f_test(x, y))
        self.f_scores[torch.isnan(self.f_scores)] = 0
        self.f_scores = self.f_scores / self.f_scores.norm(p=2)
        x = self.bias(x)
        self.S = [0]
        self.loss_fn = nn.CrossEntropyLoss()
        while len(self.S) < self.n_features + 1:
            self.train(x, y)
            input_gradient = self.average(x, y, self.n_dropouts)
            self.new = self.find(input_gradient)
            self.S.append(self.new)
        selection = self.S
        selection.remove(0)
        selection = [s - 1 for s in selection]
        return selection
