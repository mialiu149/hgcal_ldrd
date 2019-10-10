Code repository for Muon GNN

Setup:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
export CPATH=/usr/local/cuda/include:$CPATH
echo export PATH="$HOME/miniconda/bin:$PATH" >> ~/.bashrc
echo export CPATH=/usr/local/cuda/include:$CPATH >> ~/.bashrc
conda create -n torch -c pytorch Python=3.6 numpy cuda100 magma-cuda100 pytorch
conda init bash
source $HOME/.bashrc
conda activate torch
conda install pandas matplotlib jupyter nbconvert==5.4.1
conda install -c conda-forge tqdm
pip install uproot scipy sklearn --user
pip install torch-scatter torch-sparse
pip install networkx
#get this repo
git clone git@github.com:mialiu149/heptrkx-gnn-tracking.git 
#get torch geometric
for a in pytorch_cluster pytorch_spline_conv pytorch_geometric; do 
    git clone https://github.com/rusty1s/$a.git
pushd $a 
    python setup.py install
popd 
done
```
(or replace pip with the corresponding conda installation for safer compatibility)

and install pytorch geometric according to the instructions here:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

