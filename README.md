# Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data

## Abstract

Causality in time series can be difficult to determine, especially in the presence of non-linear dependencies. The concept of Granger causality helps analyze potential relationships between variables, thereby offering a method to determine whether one time series can predict—Granger cause—future values of another. Although successful, Granger causal methods still struggle with capturing long-range relations between variables. To this end, we leverage the recently successful Extended Long Short-Term Memory (xLSTM) architecture and propose Granger causal xLSTMs (GC-xLSTM). It first enforces sparsity between the time series components by using a novel dynamic lasso penalty on the initial projection. Specifically, we adaptively improve the model and identify sparsity candidates. Our joint optimization procedure then ensures that the Granger causal relations are recovered in a robust fashion. Our experimental evaluations on three datasets demonstrate the overall efficacy of our proposed GC-xLSTM model.

## Requirements

Create a conda environment from the file `environment_pt220cu121.yaml`.  
```bash
conda env create -n xlstm -f environment_pt220cu121.yaml
conda activate xlstm
```

This code has been tested with CUDA 12.1. 
The code for the xLSTM-based modules has been largely adapted from the original [xLSTM repository](https://github.com/NX-AI/xlstm).  

## Datasets

The repository contains all the datasets used in the paper, except ACATIS, which is not open-source. Two simulated datasets, the Lorenz-96 and the VAR dataset, can be created using the `GC-xLSTM/synthetic.py` python script. The other real world datasets are in the `datasets` folder. The `prepare_data.py` script contains the code to return the data, ground truth Granger causal relations (if available), and other information. For MoCap, only the processed numpy files are provided, for two actions: running and salsa dance. The original dataset can be downloaded from [CMU MoCap](http://mocap.cs.cmu.edu/).  


## Demo

When inside the `GC-xLSTM` folder, after activating the conda environment, use the `run.sh` script with a mandatory argument for the config file name. Optionally, you can specify the GPU device to use (defaults to 0). The config file must be in the `configs` folder.    
```bash
./run.sh lorenz/F40T1000.yaml 1
```

## Citation

If you find our work useful, please consider citing our paper using the following BibTeX entry:

```bibtex
@article{poonia2025grangercausality,
      title={Exploring Neural Granger Causality with xLSTMs: Unveiling Temporal Dependencies in Complex Data}, 
      author={Harsh Poonia and Felix Divo and Kristian Kersting and Devendra Singh Dhami},
      year={2025},
      eprint={2502.09981},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.09981}, 
}
```