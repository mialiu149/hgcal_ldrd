import torch
class RegressionNet(torch.nn.Module):
    def __init__(self,input_dim=3, hidden_dim=[64,64,32], output_dim=1):
        super().__init__()
        self.l1 = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim[0]),
                               torch.nn.ReLU(),
                               torch.nn.Linear(hidden_dim[0], hidden_dim[1]),
                               torch.nn.ReLU(),
                               torch.nn.Linear(hidden_dim[1], hidden_dim[2]))
        self.relu = torch.nn.ReLU()
        self.l2=torch.nn.Linear(hidden_dim[2], output_dim)

    def forward(self, X):
        return self.l2(self.relu(self.l1(X))).squeeze(-1)
