#%%
import os
import os.path as osp
import math

import numpy as np  
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset,DataLoader
from datasets.hitgraphs import HitGraphDataset
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.data import (Data, Dataset)
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)
import tqdm
import argparse

from models.gnn_geometric import GNNSegmentClassifier as Net
from models.EdgeNet import EdgeNet
from models.RegressionNet import RegressionNet

from datasets.graph import draw_sample


import awkward
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn

from training.gnn import GNNTrainer
from training.regression import RegressionTrainer

sig_weight = 1.0
bkg_weight = 1.0
batch_size = 64*4
hidden_dim = 64
n_iters = 12
nhitfeature = 6
nhits_sel = 12
lr = 1e-3
n_epoch = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

def lr_scaling(optimizer):
        from torch.optim.lr_scheduler import LambdaLR
        
        lr_type = 'linear'
        lr_warmup_epochs = 0
        
        warmup_factor = 0.
        if lr_scaling == 'linear':
            warmup_factor = 1.
        
        # LR ramp warmup schedule
        def lr_warmup(epoch, warmup_factor=warmup_factor,
                      warmup_epochs=lr_warmup_epochs):
            if epoch < warmup_epochs:
                return (1. - warmup_factor) * epoch / warmup_epochs + warmup_factor
            else:
                return 1.

        # give the LR schedule to the trainer
        return LambdaLR(optimizer, lr_warmup)
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
    true_edge = (target > threshold)
    fake_edge = (target < threshold)
    true_edge_score = output[true_edge]
    fake_edge_score = output[fake_edge]
    #factorize the plotting part
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist([true_edge_score,fake_edge_score],
    weights=[[sig_weight]*len(true_edge_score),[bkg_weight]*len(fake_edge_score)], 
    bins=100,color=['b','r'],label=['true edge','false edge'],histtype='step',fill=False)

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
    false_pos = ((output > threshold) & (target < threshold))

    print('cut', threshold,
          ': signal efficiency for true edges: ', sum(true_pos)/sum(target > threshold),
          ': fake edge ', sum(false_pos)/len(target< threshold))
    return 
# this is one way to define a network

def main(args):
    directed = False
    threshold = 0.
    path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'muon_graph_v4')   #
    #path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_mu_v0')
    full_dataset = HitGraphDataset(path, directed=directed)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    fulllen = len(full_dataset)
    tv_frac = 1.0
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])

    train_dataset = HitGraphDataset(path, directed=directed)[0:splits[0]]
    valid_dataset = HitGraphDataset(path, directed=directed)[splits[0]:splits[1]] 
    test_dataset = HitGraphDataset(path, directed=directed)[splits[1]:splits[2]]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
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
    y,pred = tester.predict(full_loader)
    make_test_plots(y,pred,threshold,osp.join(tester.output_dir,'bestmodel.pdf'))
    
    # now need to load torch orgininal dataset again to find the hits associated with the edges
    scores = []
    totalscores = []
    n_edges = []
    ntrue_edges=[]
    nhits= []
    pt_target = []
    hit_features = []

    for i in range(splits[1],splits[2]):
       data = test_dataset.get(i).to(device)
       edges_score = tester.model(data).cpu().detach().numpy()
       x= data.x.cpu().detach().numpy()
       y= data.y.cpu().detach().numpy()
       pt = data.pt
       eta = data.eta
       pt_target.append(data.pt.item(0))
       edge_index_array = np.asarray(data.edge_index.cpu().detach().numpy())
       edge_score_array = np.asarray(edges_score)
       hits_score = np.asarray((edge_index_array[0],edge_score_array))[:,edges_score>threshold]
       hits_true_score = np.asarray((edge_index_array[0],y))[:,edges_score>threshold]
       ### now calculate some hit level features
       df = pd.DataFrame(hits_score.T)
       sum_edges_score = df.groupby(0, as_index=False).sum().to_numpy()
       ### average hit score and coordinates will be used for regression, sorted by average hit score
       sum_edges_score_ave = df.groupby(0, as_index=False).mean().sort_values(1,ascending=False).to_numpy()
       x_filtered = x[sum_edges_score_ave[:,0].astype(int),64:64+nhitfeature-1]
       x_filtered = np.concatenate((x_filtered,sum_edges_score_ave[:,1][:,np.newaxis]),axis=1)
       x_filtered = x_filtered*np.array([1/1000,1/45,1/180,1/800,1,1])
       x_padded = np.zeros((nhits_sel,nhitfeature))
       if x_filtered.shape[0] < nhits_sel: x_padded[:x_filtered.shape[0],:x_filtered.shape[1]] = x_filtered
       else: x_padded = x_filtered[:nhits_sel,:]
       hit_features.append(x_padded.flatten())
       ### average number of edges, and number of average true edges.
       sum_edges = df.groupby(0, as_index=False).count().to_numpy()
       sum_true_edges = pd.DataFrame(hits_true_score.T).groupby(0, as_index=False).sum().to_numpy()
       ### save other features
       scores.append(sum_edges_score_ave)
       totalscores.append(sum_edges_score)
       n_edges.append(sum_edges)
       ntrue_edges.append(sum_true_edges)
       nhits.append(sum_edges.shape)

        # plotting:
    target = torch.Tensor(pt_target)   
    features = torch.Tensor(hit_features)
    dataset = TensorDataset(features,target)
    tv_frac = 0.8
    train_set, val_set = torch.utils.data.random_split(dataset, [math.ceil(len(dataset)*tv_frac),len(pt_target)-math.ceil(len(dataset)*tv_frac)])
    train_dataloader = DataLoader(train_set)
    val_dataloader = DataLoader(val_set)

    regressor = RegressionTrainer(real_weight=sig_weight, fake_weight=bkg_weight, 
    device=device, output_dir = os.path.abspath(os.path.join(os.path.dirname( args.model ), '..')))
    regressor.build_model(learning_rate=lr, lr_scaling=lr_scaling,input_dim=nhits_sel*nhitfeature, hidden_dim=[64,32,32])
    train_summary = regressor.train(train_dataloader, n_epochs=n_epoch, valid_data_loader=val_dataloader)
    targ,pred = regressor.predict(val_dataloader)
    print(train_summary)
    print(targ)
    print(pred)

    figs = []
    #factorize the plotting part
    fig,axes = plt.subplots(figsize=(12, 7))
    plt.plot(train_summary['train_loss'],label = 'train loss')
    plt.title('train loss')
    figs.append(fig)

    fig,axes = plt.subplots(figsize=(12, 7))
    plt.plot(train_summary['valid_loss'],label = 'valid loss')
    plt.title('validation loss')
    figs.append(fig)
    
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist([targ,pred], bins=100,range = (0,100),color=['r','b'],label=['target','pred'],histtype='step',fill=False)
    plt.title('pt distribution')
   # plt.xlim(0,100)

    figs.append(fig)
    fig,axes = plt.subplots(figsize=(12, 7))
    plt.scatter(pred,targ)
    plt.xlim(0,100)
    plt.ylim(0,100)
    figs.append(fig)
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist(np.concatenate(scores)[:,1],bins=100,color=['b'],label=['edge score'],histtype='step',fill=False)
    plt.title("Edge classifier score (per hit) on test data")
    plt.ylabel("Number of edges")
    plt.xlabel("Classifier score")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)

    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist(np.concatenate(nhits),bins=100,color=['b'],label=['Number of hits'],histtype='step',fill=False)
    plt.title("Number of hits")
    plt.ylabel("Number of events")
    plt.xlabel("Number of hits")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)

    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist(np.concatenate(totalscores)[:,1],bins=100,color=['b'],label=['Total score'],histtype='step',fill=False)
    plt.title("sum of edge classifier score per hit")
    plt.ylabel("Number of edges")
    plt.xlabel("Classifier score")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)

    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist(np.concatenate(n_edges)[:,1],bins=100,color=['b'],label=['Number of edges'],histtype='step',fill=False)
    plt.title("Number of edges per hit")
    plt.ylabel("Number of edges")
    plt.xlabel("Classifier score")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)
    fig,axes = plt.subplots(figsize=(12, 7))
    _, bins,_ = axes.hist(np.concatenate(ntrue_edges)[:,1][np.concatenate(scores)[:,1]>0.7],bins=100,color=['b'],label=['true edge number'],histtype='step',fill=False)
    plt.title("Number of true edges per hit")
    plt.ylabel("Number of edges")
    plt.xlabel("Classifier score")
    plt.legend(loc='upper left')
    plt.yscale('log')
    figs.append(fig)
  
    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages('hit_score.pdf')
    for fig in figs: 
        pdf.savefig(fig)
    pdf.close()
       #torch.save(outdata,osp.join('post_processing', 'data_{}.pt'.format(data.event_index)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("model", help="model PyTorch state dict file [*.pth]")
    args = parser.parse_args()
    main(args)
