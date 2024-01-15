import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertForMaskedLM, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import argparse
import numpy as np
import sys,os
sys.path.append(os.getcwd())
from Process.lm_loadsplits import *
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pickle

datasetname = 'fang'
n_samples = 32

train_conf, adj = pickle.load(open('data/adjs/user_t5/' + datasetname + '_nn_relations_' + str(n_samples) + '.pkl', 'rb'))
A_nn = adj.todense()
train_conf2, adj2 = pickle.load(open('data_test/' + datasetname + '_nn_relations_' + str(n_samples) + '.pkl', 'rb'))
A_nn2 = adj2.todense()

print(np.array_equal(A_nn, A_nn2))
print(np.array_equal(train_conf, train_conf2))