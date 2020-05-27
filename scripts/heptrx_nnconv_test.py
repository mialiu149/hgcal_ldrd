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
import sklearn

from training.gnn import GNNTrainer

sig_weight = 1.0
bkg_weight = 0.15
batch_size = 32
hidden_dim = 64
n_iters = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def make_test_plots(target,output,threshold, plotoutput):
    # plotting:
    figs = []
    fpr, tpr, _ = sklearn.metrics.roc_curve(np.array(target),np.array(output))
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure(figsize=(9,4))
    # Plot the ROC curve
    roc_curve,axes = plt.subplots(figsize=(12, 7))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    figs.append(roc_curve)
    predicted_edge = (output> threshold)
    true_edge = (target > threshold)
    fake_edge = (output < threshold)
    true_edge_score = output[true_edge]
    fake_edge_score = output[fake_edge]
    #factorize the plotting part
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist([true_edge_score,fake_edge_score],weights=[[sig_weight]*len(true_edge_score),[bkg_weight]*len(fake_edge_score)], bins=100,color=['b','r'],label=['true edge','false edge'],histtype='step',fill=False)
    plt.title("Edge classifier score on test data")
    plt.ylabel("Number of edges")
    plt.xlabel("Classifier score")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)
    
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(plotoutput)
    for fig in figs: 
        pdf.savefig(fig)
    pdf.close()
    # accurary
    matches = ((output > threshold) == (target > threshold))
    true_pos = ((output > threshold) & (target > threshold))
    true_neg = ((output < threshold) & (target < threshold))
    false_pos = ((output > threshold) & (target < threshold))
    false_neg = ((output < threshold) & (target > threshold))
    print('cut', threshold,
          'signal efficiency for true edges: ', true_pos,
          'fake edge ', false_pos)
    return 

def main(args):
    directed = False
    path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'muon_graph_v4_small')   #
    #path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_mu_v0')
    full_dataset = HitGraphDataset(path, directed=directed)
    fulllen = len(full_dataset)
    tv_frac = 0.2
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    test_dataset = HitGraphDataset(path, directed=directed)[splits[1]:splits[2]]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_samples = len(test_dataset)
    print('Testing with %s samples'%test_samples)

    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)

    tester = GNNTrainer(real_weight=sig_weight, fake_weight=bkg_weight, 
    device=device, output_dir = os.path.abspath(os.path.join(os.path.dirname( args.model ), '..')))
    tester.load_model('EdgeNet',input_dim=num_features,hidden_dim=hidden_dim)
    print('Model: \n%s\nParameters: %i' % (tester.model, sum(p.numel()
          for p in tester.model.parameters())))
    tester.model.load_state_dict(torch.load(args.model)['model'])
    y,pred,events = tester.predict(test_loader)
    print(y[0], pred[0], events[0])   
    # plotting:
   # make_test_plots(y,pred,0.5,osp.join(tester.output_dir,'lastmodel.pdf'))
    # now need to load torch orgininal dataset again to find the hits associated with the edges
    test = test_dataset.get(8842)
 #   print(len(pred))
 #   print(test)
    print(test_dataset.get(7075))
 #   print(test.y)
 #   print(test.edge_index)
 #   print(test.edge_index[0])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("model", help="model PyTorch state dict file [*.pth]")
    args = parser.parse_args()
    main(args)
