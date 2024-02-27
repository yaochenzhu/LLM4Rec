'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import numpy as np

def Recall_at_k(y_true, y_pred, k, agg="sum"):
    '''
        Average recall for top k recommended results.
        The training records should be set to -inf in y_pred
    '''
    batch_size = y_pred.shape[0]
    topk_idxes = np.argpartition(-y_pred, k, axis=1)[:, :k]
    y_pred_bin = np.zeros_like(y_pred, dtype=bool)
    y_pred_bin[np.arange(batch_size)[:, None], topk_idxes] = True
    y_true_bin = (y_true > 0)
    hits = np.sum(np.logical_and(y_true_bin, y_pred_bin), axis=-1).astype(np.float32)
    recalls = hits/np.minimum(k, np.sum(y_true_bin, axis=1))
    if agg == "sum":
        recall = np.sum(recalls)
    elif agg == "mean":
        recall = np.mean(recalls)
    else:
        raise NotImplementedError(f"aggregation method {agg} not defined!")
    return recall


def NDCG_at_k(y_true, y_pred, k, agg="sum"):
    '''
        Average NDCG for top k recommended results. 
        The training records should be set to -inf in y_pred
    '''

    batch_size = y_pred.shape[0]
    topk_idxes_unsort = np.argpartition(-y_pred, k, axis=1)[:, :k]
    topk_value_unsort = y_pred[np.arange(batch_size)[:, None],topk_idxes_unsort]
    topk_idxes_rel = np.argsort(-topk_value_unsort, axis=1)
    topk_idxes = topk_idxes_unsort[np.arange(batch_size)[:, None], topk_idxes_rel]
    y_true_topk = y_true[np.arange(batch_size)[:, None], topk_idxes]
    y_true_bin = (y_true > 0).astype(np.float32)
    weights = 1./np.log2(np.arange(2, k + 2))
    DCG = np.sum(y_true_topk*weights, axis=-1)
    normalizer = np.array([np.sum(weights[:int(n)]) for n in np.minimum(k, np.sum(y_true_bin, axis=-1))])
    if agg == "sum":
        NDCG = np.sum(DCG/normalizer)
    elif agg == "mean":
        NDCG = np.mean(DCG/normalizer)
    return NDCG
