#%%
import os
import os.path as osp
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.hitgraphs import HitGraphDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
import tqdm
import argparse

from models.gnn_geometric import GNNSegmentClassifier as Net
from models.EdgeNet import EdgeNet

from datasets.graph import draw_sample

import awkward
import matplotlib.pyplot as plt
import scipy.stats as stats

batch_size = 32
hidden_dim = 64
n_iters = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def test(model,loader,total):
    model.eval()
    correct = 0
    sum_loss = 0
    sum_correct = 0
    sum_truepos = 0
    sum_trueneg = 0
    sum_falsepos = 0
    sum_falseneg = 0
    sum_true = 0
    sum_false = 0
    sum_total = 0
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_target = data.y
        batch_output = model(data)
        batch_loss_item = F.binary_cross_entropy(batch_output, batch_target).item()
        t.set_description("batch loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        matches = ((batch_output > 0.5) == (batch_target > 0.5))
        true_pos = ((batch_output > 0.5) & (batch_target > 0.5))
        true_neg = ((batch_output < 0.5) & (batch_target < 0.5))
        false_pos = ((batch_output > 0.5) & (batch_target < 0.5))
        false_neg = ((batch_output < 0.5) & (batch_target > 0.5))
        sum_truepos += true_pos.sum().item()
        sum_trueneg += true_neg.sum().item()
        sum_falsepos += false_pos.sum().item()
        sum_falseneg += false_neg.sum().item()
        sum_correct += matches.sum().item()
        sum_true += batch_target.sum().item()
        sum_false += (batch_target < 0.5).sum().item()
        sum_total += matches.numel()

    print('scor', sum_correct,
          'stru', sum_true,
          'stp', sum_truepos,
          'stn', sum_trueneg,
          'sfp', sum_falsepos,
          'sfn', sum_falseneg,
          'stot', sum_total)
    return sum_loss/(i+1), sum_correct / sum_total, sum_truepos/sum_true, sum_falsepos / sum_false, sum_falseneg / sum_true, sum_truepos/(sum_truepos+sum_falsepos + 1e-6)
#%%
def main(args):
    directed = False
    path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'muon_graph_v4')
    #print(path)
    full_dataset = HitGraphDataset(path, directed=directed)
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    
    test_dataset = HitGraphDataset(path, directed=directed, pre_filter=np.arange(start=splits[1],stop=splits[2]))
    #np.arange(start=splits[2],stop=len(full_dataset)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_samples = len(test_loader)
    print(len(test_loader))
    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)
    
    model = EdgeNet(input_dim=num_features,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
    model_fname = args.model
    print('Model: \n%s\nParameters: %i' %
          (model, sum(p.numel()
                      for p in model.parameters())))
    print('Testing with %s samples'%test_samples)
    model.load_state_dict(torch.load(model_fname)['model'])

    test_loss, test_acc, test_eff, test_fp, test_fn, test_pur = test(model, test_loader, test_samples)
    print('Testing: Loss: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(test_loss, test_eff,
                                                                                                            test_fp, test_fn, test_pur))
    print(len(test_loader))
    # plotting:
    figs = []
    t = tqdm.tqdm(enumerate(test_loader),total=test_samples/batch_size)
    out = []
    y = []
    x = []
    edge_index = []
    simmatched = []
    for i,data in t:
        data = data.to(device)
        out.append(model(data).cpu().detach().numpy())
        x.append(data.x.cpu().detach().numpy())
        y.append(data.y.cpu().detach().numpy())
        edge_index.append(data.edge_index.cpu().detach().numpy())
    out = awkward.fromiter(out)
    x = awkward.fromiter(x)
    y = awkward.fromiter(y)
    edge_index = awkward.fromiter(edge_index)
    import sklearn
    fpr, tpr, _ = sklearn.metrics.roc_curve(y,out)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure(figsize=(9,4))

    # Plot the model outputs
    plt.subplot(121)
    binning=dict(bins=50, range=(0,1), histtype='bar')
    plt.hist(flat_pred[flat_y<thresh], label='fake', **binning)
    plt.hist(flat_pred[flat_y>thresh], label='true', **binning)
    plt.xlabel('Model output')
    plt.legend(loc=0)

    # Plot the ROC curve
    plt.subplot(122)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
 
    cut = 0.5
    predicted_edge = (out > cut)
    true_edge = (y > 0.5)
    fake_edge = (y < 0.5)
    node_layer = x[-1]
    true_edge_score = out[true_edge]
    fake_edge_score = out[fake_edge]
    predicted_connected_node_indices = awkward.JaggedArray.concatenate([edge_index[:,0][predicted_edge], edge_index[:,1][predicted_edge]], axis=1)
    predicted_connected_node_indices = awkward.fromiter(map(np.unique, predicted_connected_node_indices))
    true_connected_node_indices = awkward.JaggedArray.concatenate([edge_index[:,0][true_edge],edge_index[:,1][true_edge]], axis=1)
    true_connected_node_indices = awkward.fromiter(map(np.unique, true_connected_node_indices))
    #def buildtracks(edges):
        
    #   return tracks
    #factorize the plotting part
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist([true_edge_score.flatten(),fake_edge_score.flatten()],weights=[[1]*len(true_edge_score.flatten()),[0.15]*len(fake_edge_score.flatten())], bins=100,color=['b','r'],label=['true edge','false edge'],histtype='step',fill=False)

    plt.title("Edge classifier score on test data")
    plt.ylabel("Number of edges")
    plt.xlabel("Classifier score")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)
    
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages("test_plots.pdf")
    for fig in figs: 
        pdf.savefig(fig)
    pdf.close()

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("model", help="model PyTorch state dict file [*.pth]")
    args = parser.parse_args()
    main(args)
