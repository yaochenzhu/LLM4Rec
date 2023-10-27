import re
import os
import sys
import pickle
import fsspec
import random
import argparse
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from transformers import GPT2Model, GPT2Config
from transformers import GPT2Tokenizer

sys.path.append("libs")
from tokenizer import TokenizerWithUserItemIDTokensBatch

from data import UserItemContentGPTDatasetBatch
from data import RecommendationGPTTrainGeneratorBatch
from data import RecommendationGPTTestGeneratorBatch

from model import GPT4RecommendationBaseModel
from model import ContentGPTForUserItemWithLMHeadBatch
from model import CollaborativeGPTwithItemRecommendHead
from util import Recall_at_k, NDCG_at_k
    
def save_local(remote_path, local_path, remote_mode, local_mode):
    '''
        Save the remote file in remote_path
        to the local_path...
    '''
    with fsspec.open(remote_path, remote_mode) as f:
        content = f.read()
    with fsspec.open(local_path, local_mode) as f:
        f.write(content)


def save_remote(local_path, remote_path, local_mode, remote_mode):
    '''
        Save the local file in local_path
        to the remote_path...
    '''
    with fsspec.open(local_path, local_mode) as f:
        content = f.read()
    with fsspec.open(remote_path, remote_mode) as f:
        f.write(content)

server_root = "hdfs://llm4rec"
local_root = "tmp"
if not os.path.exists(local_root):
    os.makedirs(local_root, exist_ok=True)

_config = {
    "activation_function": "gelu_new",
    "architectures": [
    "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
    "text-generation": {
        "do_sample": True,
        "max_length": 50
    }
    },
    "vocab_size": 50257
}

def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str,
        help="specify the dataset for experiment")
    args = parser.parse_args()
    
    dataset = args.dataset
    
    print("-----Current Setting-----")
    print(f"dataset: {dataset}")
    print(f"lambda_V: {args.lambda_V}")
    
    # Define the device
    device = "cuda"
    
    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")
    
    '''
        Get the basic information of the dataset
    '''
    print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(server_root, "dataset", dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
        
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    print(f"num_users: {num_users}")
    print(f"num_items: {num_items}")
    print("-----End Obtaining Dataset Info-----\n")


    '''
        Obtain the tokenizer with user/item tokens
    '''
    print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = os.path.join(server_root, "model", "pretrained", "tokenizer")
    print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    remote_vocab_file = os.path.join(tokenizer_root, "vocab_file.json")
    remote_merges_file = os.path.join(tokenizer_root, "merges.txt")
    vocab_file = os.path.join(local_root, "vocab_file.json")
    merges_file = os.path.join(local_root, "merges.txt")

    save_local(remote_vocab_file, vocab_file, "r", "w")
    save_local(remote_merges_file, merges_file, "r", "w")
        
    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file, 
                                                   merges_file,
                                                   num_users,
                                                   num_items)
    print("Success!")
    print("-----End Obtaining the Tokenizer-----\n")


    '''
        Obtain the testing data generator
    '''
    print("-----Begin Obtaining the Collaborative Data Generator-----")
    remote_train_mat_path = os.path.join(data_root, "train_matrix.npz")
    local_train_mat_path = os.path.join(local_root, "train_matrix.npz")
    print(f"Loading data from {remote_train_mat_path}...")
    save_local(remote_train_mat_path, local_train_mat_path, "rb", "wb")
    
    remote_test_mat_path = os.path.join(data_root, "test_matrix.npz")
    local_test_mat_path = os.path.join(local_root, "test_matrix.npz")
    save_local(remote_test_mat_path, local_test_mat_path, "rb", "wb")
    
    # Get the testing data generator
    train_mat = load_npz(local_train_mat_path)
    test_mat = load_npz(local_test_mat_path)
    test_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, test_mat)

    print("Success!")
    print("-----End Obtaining the Collaborative Data Generator-----\n")


    '''
        Extend the config of the original GPT model
    '''
    print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items
    print("Success!")
    print("-----End Setting Up the Config-----\n")


    '''
        Instantiate the pretrained GPT2 model
    '''
    print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2Model(config)
    pretrained_root = os.path.join(server_root, "model", "pretrained")
    print(f"Loading pretrained weights from {pretrained_root}...")
    remote_pretrained_weights_path = os.path.join(pretrained_root, "gpt2", "pytorch_model.bin")
    local_pretrained_weights_path = os.path.join(local_root, "gpt2", "pytorch_model.bin")
    save_local(remote_pretrained_weights_path, local_pretrained_weights_path, "rb", "wb")
    gpt2model.load_state_dict(torch.load(local_pretrained_weights_path), strict=False)
    print("Success!")
    print("-----End Instantiating the Pretrained GPT Model-----\n")


    '''
        Instantiate the GPT for recommendation content model
    '''
    print("-----Begin Instantiating the Content GPT Model-----")
    base_model = GPT4RecommendationBaseModel(config, gpt2model)

    pretrained_root = os.path.join(server_root, "model", dataset, "rec")
    remote_pretrained_user_emb_path = os.path.join(pretrained_root, f"user_embeddings_{args.lambda_V}.pt") 
    remote_pretrained_item_emb_path = os.path.join(pretrained_root, f"item_embeddings_{args.lambda_V}.pt") 
    local_pretrained_user_emb_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}.pt")
    local_pretrained_item_emb_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}.pt")
    
    save_local(remote_pretrained_user_emb_path, local_pretrained_user_emb_path, "rb", "wb")
    save_local(remote_pretrained_item_emb_path, local_pretrained_item_emb_path, "rb", "wb")    

    base_model.user_embeddings.load_state_dict(
        torch.load(local_pretrained_user_emb_path, map_location=device))
    print("Load pretrained user embeddings: Success!")
    base_model.item_embeddings.load_state_dict(
        torch.load(local_pretrained_item_emb_path, map_location=device))
    print("Load pretrained item embeddings: Success!")

    rec_model = CollaborativeGPTwithItemRecommendHead(config, base_model)
    print("Success!")
    print("-----End Instantiating the Content GPT Model-----\n")

    
    '''
        Create a data sampler for distributed training
    '''
    print("-----Begin Creating the DataLoader-----")

    # Create the testing data loader
    # Note that we only do the testing in the main process!
    batch_size = 256
    test_data_loader = DataLoader(test_data_gen, 
                                  batch_size=batch_size, 
                                  collate_fn=test_data_gen.collate_fn)
    print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    
    # Set the model to evaluation mode
    rec_model.eval()  
    cur_recall_20 = 0
    cur_recall_40 = 0
    cur_NDCG_100 = 0

    with torch.no_grad():
        for input_ids, train_mat, target_mat, attention_mask in test_data_loader:
            # Move tensors to the correct device
            input_ids = input_ids.to(device)
            train_mat = train_mat.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)

            # Get item scores and rank them
            rec_loss, item_scores = rec_model(input_ids, 
                                                target_mat, 
                                                attention_mask)
            
            # Set score of interacted items to the lowest
            item_scores[train_mat > 0] = -float("inf")  

            # Calculate Recall@K and NDCG@K for each user
            target_mat = target_mat.cpu().numpy()
            item_scores = item_scores.cpu().numpy()
            cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
            cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
            cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

    # Calculate average Recall@K and NDCG@K for the validation set
    cur_recall_20 /= len(test_data_gen)
    cur_recall_40 /= len(test_data_gen)
    cur_NDCG_100 /= len(test_data_gen)
    
    print(f"Final Testing Results:")
    print(f"Recall@20: {cur_recall_20:.4f}")
    print(f"Recall@40: {cur_recall_40:.4f}")
    print(f"NDCG@100: {cur_NDCG_100:.4f}")
    
    results_path = os.path.join(pretrained_root, f"results_{args.lambda_V}.txt")
    with fsspec.open(results_path, "w") as f:
        f.write("Recall@20,Recall@40,NDCG@100\n")
        f.write(f"{cur_recall_20:.4f},{cur_recall_40:.4f},{cur_NDCG_100:.4f}")


if __name__ == "__main__":
    main()