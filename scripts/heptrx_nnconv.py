#%%
import os
import os.path as osp
import math
import glob
import numpy as np
import torch

#from torch.utils.data import Dataset, DataLoader
torch.cuda.is_available()
torch.version.cuda
#%%
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

from datasets.hitgraphs import HitGraphDataset
from models.EdgeNet import EdgeNet

import tqdm
import argparse
directed = False
fulldata = True
sig_weight = 1.0
bkg_weight = 0.15
batch_size = 64
n_epochs = 10
lr = 0.01
hidden_dim = 64
n_iters = 12

from training.gnn import GNNTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)

import logging

import awkward
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn  

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
          'signa efficiency for true edges: ', true_pos,
          'fake edge ', false_pos)
    return 

def main(args):    
   # path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'single_mu_v0')
    path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], 'muon_graph_v4_small')
        
    full_dataset = HitGraphDataset(path, directed=directed)
    fulllen = 1000
    if fulldata: fulllen=len(full_dataset)
    # splitting datasets
    tv_frac = 0.2
    tv_num = math.ceil(int(fulllen)*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])   

    print("train, validation, testing splitting : ",fulllen, splits)

    train_dataset = HitGraphDataset(path, directed=directed)[0:splits[0]]
    valid_dataset = HitGraphDataset(path, directed=directed)[splits[0]:splits[1]]  
    test_dataset = HitGraphDataset(path, directed=directed)[splits[1]:splits[2]] 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)
    print("Number of training samples   : ",train_samples)
    print("Number of validation samples : ",valid_samples)
    print("Number of testing samples    :  ",test_samples)

    d = full_dataset
    num_features = d.num_features
    num_classes = d[0].y.max().item() + 1 if d[0].y.dim() == 1 else d[0].y.size(1)

    trainer = GNNTrainer(real_weight=sig_weight, fake_weight=bkg_weight, 
                         output_dir=args.output_dir, device=device)

    trainer.logger.setLevel(logging.DEBUG)
    strmH = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    strmH.setFormatter(formatter)
    trainer.logger.addHandler(strmH)
        
    #example lr scheduling definition
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
    
    trainer.build_model(name='EdgeNet', loss_func='binary_cross_entropy',
                        optimizer='Adam', learning_rate=0.01, lr_scaling=lr_scaling,
                        input_dim=num_features, hidden_dim=hidden_dim, n_iters=n_iters)
    
    print('made the hep.trkx trainer!')
    train_summary = trainer.train(train_loader, n_epochs, valid_data_loader=valid_loader)
    print(train_summary)

    # plot for the last epoch
    y,pred = trainer.predict(test_loader)
    make_test_plots(y,pred,0.5,osp.join(trainer.output_dir,'lastmodel.pdf'))

    # plot for the best model
    output_checkpoint = glob.glob(os.path.join(trainer.output_dir, 'checkpoints')+'/*.tar')
    bestmodel_path = [i for i in output_checkpoint if 'best' in i][0]
    trainer.model.load_state_dict(torch.load(bestmodel_path)['model'])
    y,pred = trainer.predict(test_loader)
    make_test_plots(y,pred,0.5,osp.join(trainer.output_dir,'bestmodel.pdf'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="output directory to save training summary")
    args = parser.parse_args()
    main(args)
