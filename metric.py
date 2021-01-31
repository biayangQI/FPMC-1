'''
Author: lqyang
Date: 2021-01-25 15:20:00
LastEditors: lqyang
LastEditTime: 2021-01-25 17:11:08
FilePath: \session_related\FPMC-1\metric.py
Dec: 
from DemandRS
'''
import torch
import numpy as np

def multiply_list(a:list)->list:
    """
    Multiply each element in the list by 100
    """
    return [i*100 for i in a]

def evaluate(scores, ground_truth, Ks=[10,20,50], Ks_auc=[50,100,200,500]):
    '''
    Evaluates the model using Recall@K, MRR@K scores. For FPMC method, one batch is the whole data.
    Args:
        scores: scores, torch.Tensor, batch_size * n_items, the predicted scores for the next items.
        ground_truth: torch.Tensor, batch_size
        Ks: list, dtype=int, top@k for k in Ks, such as [10.20.50,100]
    Returns:
    metrics, dict, {'recall': [recall@10, recall@20, recall@50, recall@100], 'mrr': [mrr@10, mrr@20, mrr@50, mrr@100]}
    '''
    metrics = {}
    metrics['auc'] = multiply_list(get_auc(scores,ground_truth,Ks_auc))
    recalls, mrrs, ndcgs = [], [], []
    for k in Ks:
        _, indices = torch.topk(scores, k=k, dim=-1)  # return（values=tensor[,...,], indices=tensor[,...,]）
        targets = ground_truth.view(-1, 1).expand_as(indices)

        recall = get_recall(indices, targets)
        mrr = get_mrr(indices, targets)
        recalls.append(recall)
        mrrs.append(mrr)

        NDCG_k = NDCG(k)
        ndcg = NDCG_k(indices,targets)
        ndcgs.append(ndcg)

    metrics['recall'] = multiply_list(recalls)
    metrics['mrr'] = multiply_list(mrrs)
    metrics['ndcg'] = multiply_list(ndcgs)
    
    return metrics['recall'], metrics['mrr'], metrics['ndcg'], metrics['auc']


def get_recall(indices,ground_truth):
    """
    calculate the precision top@k of a batch data
    Args:
        indices: batch_size * k,  top-k indices predicted by the model
        ground_truth: target item id, len() = batch_size, actual target indices
        k: int, top@k
    Returns:
        recall (float): the recall score of a batch data
    """
    hits = (ground_truth==indices).nonzero()
    if len(hits) == 0:
        return 0
    else:
        recall = float(len(hits))/ground_truth.size(0)
    return recall


def get_mrr(indices, ground_truth):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices: batch_size * k,  top-k indices predicted by the model
        ground_truth: target item id, len() = batch_size, actual target indices
        k: int, top@k
    Returns:
        mrr (float): the mrr score of a batch data
    """
    hits = (ground_truth == indices).nonzero()
    ranks = hits[:,1]+1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / ground_truth.size(0)
    return mrr.item()


class NDCG(object):
    def __init__(self, k, idcg_sum_k=False):
        super(NDCG, self).__init__()
        self.topk = k
        self.idcg_sum_k = idcg_sum_k
        self.idcg = self.IDCG()
        self.is_hit = torch.zeros(self.topk, dtype=torch.float)

    def DCG(self, indices, ground_truth, device): # 在GPU上计算
        hits = (ground_truth == indices).nonzero()
        ranks = hits[:, 1]
        ranks = ranks.float()
        dcg_u = torch.reciprocal(torch.log2(ranks+2).to(device))
        dcg = torch.sum(dcg_u).data / ground_truth.size(0)
        return dcg.item()

    def IDCG(self):  #在cpu上计算
        if self.idcg_sum_k:
            #print('NDCG@%d: calculate the sum of position form 1 to topk'%(self.topk))
            hit = torch.zeros(self.topk, dtype=torch.float)
            hit[:] = 1
            hit = hit / torch.log2(torch.arange(2, self.topk + 2, dtype=torch.float))
            return hit.sum(-1)
        else:
            #print('NDCG@%d: only calculate the first position ' % (self.topk))
            return 1 / torch.log2(torch.tensor(2.0))

    def __call__(self, indices, ground_truth): # 在GPU上计算
        device = indices.device
        dcg = self.DCG(indices, ground_truth, device)
        ndcg = dcg / self.idcg.to(device)
        return ndcg


def get_auc(scores, ground_truth, Ks_auc = [100,200,500]):
    """

    Args:
        scores: torch.tensor, batch_size * (n_item+1)
        ground_truth: torch.tensor, batch_size
        n_items: int, the number of n_item （include 0）in the whole data
        Ks_auc: list, such as for k in [100,200,500], where k = N ,(N is the number of negative examples)
    Returns:
        auc(float): the auc value of a batch data
    """
    device = scores.device
    batch_size, n_items = scores.shape
    Ks_auc_max = np.max(Ks_auc)

    index_col = torch.randint(low=0, high=n_items, size=(batch_size, Ks_auc_max)).to(device)
    ground_truth_ = ground_truth.view(-1, 1).expand(-1, Ks_auc_max)
    index_col = torch.where(index_col == ground_truth_, ((ground_truth_ //2) +1).long(), index_col)
    index_row = torch.arange(batch_size).view(-1, 1).expand(-1, Ks_auc_max).to(device)

    target_scores = scores[torch.arange(batch_size).to(device), ground_truth].view(-1,1)
    aucs= []
    for k in Ks_auc:
        sub_scores = scores[index_row[:,:k], index_col[:,:k]]
        sub_scores = torch.cat((sub_scores, target_scores), dim=1)
        _, idx = sub_scores.sort(dim=1, descending=True)
        _, rank = idx.sort(dim=1) # get the target id decreasing position
        rank = rank[:,-1].float() # batch_size
        auc_k = torch.mean(1-rank/k)
        aucs.append(auc_k.item())
    return aucs
