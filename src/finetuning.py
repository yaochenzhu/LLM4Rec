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

from accelerate import Accelerator

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
    # Define the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
        help="specify the dataset for experiment")
    parser.add_argument("--lambda_V", type=str,
        help="specify the dataset for experiment")
    args = parser.parse_args()
    
    dataset = args.dataset
    lambda_V = float(args.lambda_V)
    
    accelerator.print("-----Current Setting-----")
    accelerator.print(f"dataset: {dataset}")
    accelerator.print(f"lambda_V: {args.lambda_V}")
    
    # Define the number of GPUs to be used
    num_gpus = torch.cuda.device_count()
    accelerator.print(f"num_gpus: {num_gpus}")
    
    '''
        Get the basic information of the dataset
    '''
    accelerator.print("-----Begin Obtaining Dataset Info-----")
    data_root = os.path.join(server_root, "dataset", dataset)
    meta_path = os.path.join(data_root, "meta.pkl")

    with fsspec.open(meta_path, "rb") as f:
        meta_data = pickle.load(f)
        
    num_users = meta_data["num_users"]
    num_items = meta_data["num_items"]
    accelerator.print(f"num_users: {num_users}")
    accelerator.print(f"num_items: {num_items}")
    accelerator.print("-----End Obtaining Dataset Info-----\n")


    '''
        Obtain the tokenizer with user/item tokens
    '''
    accelerator.print("-----Begin Obtaining the Tokenizer-----")
    tokenizer_root = os.path.join(server_root, "model", "pretrained", "tokenizer")
    accelerator.print(f"Loading pretrained tokenizer from {tokenizer_root}...")
    remote_vocab_file = os.path.join(tokenizer_root, "vocab_file.json")
    remote_merges_file = os.path.join(tokenizer_root, "merges.txt")
    vocab_file = os.path.join(local_root, "vocab_file.json")
    merges_file = os.path.join(local_root, "merges.txt")

    if accelerator.is_main_process:
        save_local(remote_vocab_file, vocab_file, "r", "w")
        save_local(remote_merges_file, merges_file, "r", "w")
    accelerator.wait_for_everyone()
        
    tokenizer = TokenizerWithUserItemIDTokensBatch(vocab_file, 
                                                   merges_file,
                                                   num_users,
                                                   num_items)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Tokenizer-----\n")
    
    
    '''
        Define the review data generator
    '''
    accelerator.print("-----Begin Obtaining the Review Data Generator-----")
    review_path = os.path.join(data_root, "user_item_texts", "review.pkl")
    accelerator.print(f"Loading data from {review_path}...")
    review_data_gen = UserItemContentGPTDatasetBatch(tokenizer, review_path)
    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Review Data Generator-----\n")


    '''
        Obtain the training/validation data generator
    '''
    accelerator.print("-----Begin Obtaining the Collaborative Data Generator-----")
    remote_train_mat_path = os.path.join(data_root, "train_matrix.npz")
    local_train_mat_path = os.path.join(local_root, "train_matrix.npz")
    accelerator.print(f"Loading data from {remote_train_mat_path}...")
    if accelerator.is_main_process:
        save_local(remote_train_mat_path, local_train_mat_path, "rb", "wb")

    remote_val_mat_path = os.path.join(data_root, "val_matrix.npz")
    local_val_mat_path = os.path.join(local_root, "val_matrix.npz")
    if accelerator.is_main_process:
        save_local(remote_val_mat_path, local_val_mat_path, "rb", "wb")
    accelerator.wait_for_everyone()
    
    # Get the training data generator
    train_mat = load_npz(local_train_mat_path)
    train_data_gen = RecommendationGPTTrainGeneratorBatch(tokenizer, train_mat)

    # Get the validation data generator
    val_mat = load_npz(local_val_mat_path)
    val_data_gen = RecommendationGPTTestGeneratorBatch(tokenizer, train_mat, val_mat)

    accelerator.print("Success!")
    accelerator.print("-----End Obtaining the Collaborative Data Generator-----\n")


    '''
        Extend the config of the original GPT model
    '''
    accelerator.print("-----Begin Setting Up the Config-----")
    config = GPT2Config(**_config)
    config.num_users = num_users
    config.num_items = num_items
    accelerator.print("Success!")
    accelerator.print("-----End Setting Up the Config-----\n")


    '''
        Instantiate the pretrained GPT2 model
    '''
    accelerator.print("-----Begin Instantiating the Pretrained GPT Model-----")
    gpt2model = GPT2Model(config)
    pretrained_root = os.path.join(server_root, "model", "pretrained")
    accelerator.print(f"Loading pretrained weights from {pretrained_root}...")
    remote_pretrained_weights_path = os.path.join(pretrained_root, "gpt2", "pytorch_model.bin")
    local_pretrained_weights_path = os.path.join(local_root, "gpt2", "pytorch_model.bin")
    if accelerator.is_main_process:
        save_local(remote_pretrained_weights_path, local_pretrained_weights_path, "rb", "wb")
    accelerator.wait_for_everyone()
    gpt2model.load_state_dict(torch.load(local_pretrained_weights_path), strict=False)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Pretrained GPT Model-----\n")
    
    
    '''
        Instantiate the GPT for recommendation content model
    '''
    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    content_base_model = GPT4RecommendationBaseModel(config, gpt2model)
    
    pretrained_root = os.path.join(server_root, "model", dataset, "content")
    remote_pretrained_user_emb_path = os.path.join(pretrained_root, f"user_embeddings_{args.lambda_V}.pt") 
    remote_pretrained_item_emb_path = os.path.join(pretrained_root, f"item_embeddings_{args.lambda_V}.pt") 
    local_pretrained_user_emb_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}.pt")
    local_pretrained_item_emb_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}.pt")
    
    if accelerator.is_main_process:
        save_local(remote_pretrained_user_emb_path, local_pretrained_user_emb_path, "rb", "wb")
        save_local(remote_pretrained_item_emb_path, local_pretrained_item_emb_path, "rb", "wb")    
    accelerator.wait_for_everyone()

    content_base_model.user_embeddings.load_state_dict(
        torch.load(local_pretrained_user_emb_path, map_location=device))
    accelerator.print("Load pretrained user embeddings: Success!")
    content_base_model.item_embeddings.load_state_dict(
        torch.load(local_pretrained_item_emb_path, map_location=device))
    accelerator.print("Load pretrained item embeddings: Success!")
    
    content_model = ContentGPTForUserItemWithLMHeadBatch(config, content_base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")
    

    '''
        Instantiate the GPT for recommendation model
    '''
    accelerator.print("-----Begin Instantiating the Content GPT Model-----")
    base_model = GPT4RecommendationBaseModel(config, gpt2model)

    pretrained_root = os.path.join(server_root, "model", dataset, "collaborative")
    remote_pretrained_user_emb_path = os.path.join(pretrained_root, f"user_embeddings_{args.lambda_V}.pt") 
    remote_pretrained_item_emb_path = os.path.join(pretrained_root, f"item_embeddings_{args.lambda_V}.pt") 
    local_pretrained_user_emb_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}.pt")
    local_pretrained_item_emb_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}.pt")
    
    if accelerator.is_main_process:
        save_local(remote_pretrained_user_emb_path, local_pretrained_user_emb_path, "rb", "wb")
        save_local(remote_pretrained_item_emb_path, local_pretrained_item_emb_path, "rb", "wb")    
    accelerator.wait_for_everyone()

    base_model.user_embeddings.load_state_dict(
        torch.load(local_pretrained_user_emb_path, map_location=device))
    accelerator.print("Load pretrained user embeddings: Success!")
    base_model.item_embeddings.load_state_dict(
        torch.load(local_pretrained_item_emb_path, map_location=device))
    accelerator.print("Load pretrained item embeddings: Success!")

    rec_model = CollaborativeGPTwithItemRecommendHead(config, base_model)
    accelerator.print("Success!")
    accelerator.print("-----End Instantiating the Content GPT Model-----\n")


    '''
        Freeze the parameters of the pretrained GPT2 for content model
    '''
    for name, param in rec_model.named_parameters():
        # we allow only user/item token embeddings to be trained
        if ('user_embeddings' not in name) and \
           ('item_embeddings' not in name):
            param.requires_grad = False

    accelerator.print("-----Trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))
    
    accelerator.print("\n-----Non-trainable Parameters-----")
    for name, param in rec_model.named_parameters():
        if not param.requires_grad:
            accelerator.print("{} : {}".format(name, param.shape))

    '''
        Set up the training details
    '''
    accelerator.print("-----Begin Setting Up the Training Details-----")
    learning_rate = 1e-4
    batch_size = 20
    val_batch_size = 256
    num_epochs = 150


    '''
        Create a data sampler for distributed training
    '''
    accelerator.print("-----Begin Creating the DataLoader-----")

    # Create the training data loader
    train_data_loader = DataLoader(train_data_gen, 
                                   batch_size=batch_size, 
                                   collate_fn=train_data_gen.collate_fn)

    # Create the validation data loader
    # Note that we only do the validation in the main process!
    val_data_loader = DataLoader(val_data_gen, 
                                 batch_size=val_batch_size, 
                                 collate_fn=val_data_gen.collate_fn)
    
    # Create the review data loader with the custom collate_fn
    review_data_loader = DataLoader(review_data_gen, 
                                    batch_size=batch_size, 
                                    collate_fn=review_data_gen.collate_fn)
    accelerator.print("-----End Creating the DataLoader-----\n")

    # Set the model to the training mode
    rec_model.to(device)
    
    content_model.train()
    content_model.to(device)

    # Obtain the optimizer
    optimizer = optim.Adam(rec_model.parameters(), 
                           lr=learning_rate)
    
    review_optimizer = optim.Adam(content_model.parameters(), 
                                  lr=learning_rate)

    # Parallel model, optimizer and data loader with accelerator
    rec_model, optimizer, train_data_loader = accelerator.prepare(
        rec_model, optimizer, train_data_loader
    )
    
    content_model, review_optimizer, review_data_loader = accelerator.prepare(
        content_model, review_optimizer, review_data_loader
    )

    # Initialize best_loss with infinity
    review_best_loss = float('inf')
    best_val_rec_loss = float('inf')
    best_recall_20 = -float('inf')
    best_recall_40 = -float('inf')
    best_NDCG_100 = -float('inf')
    best_sum = -float('inf')

    # The place to save the recommendation model weights
    rec_model_root = os.path.join(server_root, "model", dataset, "rec")
    accelerator.print(f"Weights will be saved to {rec_model_root}!")
    
    # The place to save the content model weights
    content_model_root = os.path.join(server_root, "model", dataset, "content")
    accelerator.print(f"Weights will be saved to {content_model_root}!")
    accelerator.print("-----End Setting Up the Training Details-----\n")

    '''
        Define the pretraining loop for the content GPT
    '''
    accelerator.print("-----Begin Rec GPT Pretraining Loop-----")
    for epoch in range(num_epochs):
        # Set the model to the training mode
        rec_model.train()
        train_rec_loss = 0
        regularize_total_loss = 0 
        
        # Initialize tqdm progress bar
        #progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch + 1}", ncols=80)
                            #disable=not accelerator.is_local_main_process, ncols=80)
        #for input_ids, target_mat, attention_mask, input_ids_main in progress_bar:
        for input_ids, target_mat, attention_mask, input_ids_main in train_data_loader:
            optimizer.zero_grad()

            input_ids = input_ids.to(device)
            target_mat = target_mat.to(device)
            attention_mask = attention_mask.to(device)
            input_ids_main = input_ids_main.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                content_embeds = torch.cat(
                    (accelerator.unwrap_model(content_model).base_model.embed(input_ids),
                     accelerator.unwrap_model(content_model).base_model.embed(input_ids_main)),
                    axis=1
                ).to(device)

            # Forward pass
            outputs = rec_model(input_ids, 
                                target_mat, 
                                attention_mask=attention_mask,
                                regularize=True,
                                lambda_V=lambda_V,
                                main_ids=input_ids_main,
                                content_embeds=content_embeds)
            rec_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(rec_loss)
            optimizer.step()

            train_rec_loss += rec_loss.item()
            regularize_total_loss += regularize_loss.item()
            #progress_bar.set_postfix({"Rec Loss": rec_loss.item()})

        # Gather the multinomial recommendation loss from different device
        thread_train_rec_loss = torch.tensor([train_rec_loss / len(train_data_loader)]).to(device)
        gathered_train_rec_loss = accelerator.gather(thread_train_rec_loss)
        train_rec_loss = torch.mean(gathered_train_rec_loss)
        accelerator.print(f"Epoch {epoch + 1} - Rec Loss: {train_rec_loss:.4f}")

        # Gather the regularize loss from difference device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(train_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Set the model to evaluation mode
        rec_model.eval()  
        val_rec_loss = 0
        cur_recall_20 = 0
        cur_recall_40 = 0
        cur_NDCG_100 = 0

        accelerator.wait_for_everyone()
        with torch.no_grad():
            for input_ids, train_mat, target_mat, attention_mask in val_data_loader:
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
                val_rec_loss += rec_loss.item()
                cur_recall_20 += Recall_at_k(target_mat, item_scores, k=20, agg="sum")
                cur_recall_40 += Recall_at_k(target_mat, item_scores, k=40, agg="sum")
                cur_NDCG_100 += NDCG_at_k(target_mat, item_scores, k=100, agg="sum")

        # Calculate average Recall@K and NDCG@K for the validation set
        val_rec_loss /= len(val_data_loader)
        cur_recall_20 /= len(val_data_gen)
        cur_recall_40 /= len(val_data_gen)
        cur_NDCG_100 /= len(val_data_gen)
        cur_sum = cur_recall_20 + cur_recall_40 + cur_NDCG_100
    
        # Update the best metrics
        if val_rec_loss < best_val_rec_loss:
            best_val_rec_loss = val_rec_loss
        if cur_recall_20 > best_recall_20:
            best_recall_20 = cur_recall_20
        if cur_recall_40 > best_recall_40:
            best_recall_40 = cur_recall_40
        if cur_NDCG_100 > best_NDCG_100:
            best_NDCG_100 = cur_NDCG_100
        if cur_sum > best_sum:
            best_sum = cur_sum
            if accelerator.is_main_process:
                # Save user embeddings in the main process
                local_user_emb_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}.pt")
                remote_user_emb_path = os.path.join(rec_model_root, f"user_embeddings_{args.lambda_V}.pt")
                torch.save(accelerator.unwrap_model(rec_model).base_model.user_embeddings.state_dict(), 
                            local_user_emb_path)
                save_remote(local_user_emb_path, remote_user_emb_path, "rb", "wb")

                # Save item embeddings in the main process
                local_item_emb_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}.pt")
                remote_item_emb_path = os.path.join(rec_model_root, f"item_embeddings_{args.lambda_V}.pt")
                torch.save(accelerator.unwrap_model(rec_model).base_model.item_embeddings.state_dict(), 
                            local_item_emb_path)
                save_remote(local_item_emb_path, remote_item_emb_path, "rb", "wb")

        accelerator.print(f"Best model saved to {rec_model_root}")
        accelerator.print(f"Train Rec Loss: {train_rec_loss:.4f}")
        accelerator.print(f"Val Rec Loss: {val_rec_loss:.4f} / Best Val Rec Loss: {best_val_rec_loss:.4f}")
        accelerator.print(f"Cur Recall@20: {cur_recall_20:.4f} / Best Recall@20: {best_recall_20:.4f}")
        accelerator.print(f"Cur Recall@40: {cur_recall_40:.4f} / Best Recall@40: {best_recall_40:.4f}")
        accelerator.print(f"Cur NDCG@100: {cur_NDCG_100:.4f} / Best NDCG@100: {best_NDCG_100:.4f}")    
    
        review_total_loss = 0
        regularize_total_loss = 0
        
        #progress_bar = tqdm(review_data_loader, desc=f"Epoch {epoch + 1}",
        #                    disable=not accelerator.is_local_main_process, ncols=100)
        for input_ids_prompt, input_ids_main, attention_mask in review_data_loader:
        #for input_ids_prompt, input_ids_main, attention_mask in progress_bar:
            review_optimizer.zero_grad()

            input_ids_prompt = input_ids_prompt.to(device)
            input_ids_main = input_ids_main.to(device)
            attention_mask = attention_mask.to(device)

            accelerator.wait_for_everyone()
            with torch.no_grad():
                rec_embeds = accelerator.unwrap_model(rec_model).\
                             base_model.embed(input_ids_prompt).to(device)
                
            # Forward pass of the content GPT
            outputs = content_model(input_ids_prompt, 
                                    input_ids_main, 
                                    labels_main=input_ids_main,
                                    attention_mask=attention_mask,
                                    regularize=True,
                                    lambda_V=lambda_V,
                                    collaborative_embeds=rec_embeds)
            review_loss = outputs[0]
            regularize_loss = outputs[1]

            # Backward pass and optimization
            accelerator.backward(review_loss)
            review_optimizer.step()

            review_total_loss += review_loss.item()
            regularize_total_loss += regularize_loss.item()
            # progress_bar.set_postfix({"Review Loss": review_loss.item(),
            #                           "Regularize Loss": regularize_loss.item()})

        # Gather the content LM loss from different device
        thread_review_average_loss = torch.tensor([review_total_loss / len(review_data_loader)]).to(device)
        gathered_review_average_loss = accelerator.gather(thread_review_average_loss)
        review_average_loss = torch.mean(gathered_review_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Review Average Loss: {review_average_loss:.4f}")
        
        # Gather the regularize loss from different device
        thread_regularize_average_loss = torch.tensor([regularize_total_loss / len(review_data_loader)]).to(device)
        gathered_regularize_average_loss = accelerator.gather(thread_regularize_average_loss)
        regularize_average_loss = torch.mean(gathered_regularize_average_loss)
        accelerator.print(f"Epoch {epoch + 1} - Average Regularize Loss: {regularize_average_loss:.4f}")

        # Check if the current loss is better than the best_loss
        accelerator.wait_for_everyone()
        if review_average_loss < review_best_loss:
            review_best_loss = review_average_loss

            # Save user embeddings in the main process
            user_emb_local_path = os.path.join(local_root, f"user_embeddings_{args.lambda_V}.pt")
            user_emb_remote_path = os.path.join(content_model_root, f"user_embeddings_{args.lambda_V}.pt") 
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).base_model.user_embeddings.state_dict(), 
                           user_emb_local_path)
                save_remote(user_emb_local_path, user_emb_remote_path, "rb", "wb")
            
            # Save item embeddings in the main process
            item_emb_local_path = os.path.join(local_root, f"item_embeddings_{args.lambda_V}.pt")
            item_emb_remote_path = os.path.join(content_model_root, f"item_embeddings_{args.lambda_V}.pt")
            if accelerator.is_main_process:
                torch.save(accelerator.unwrap_model(content_model).base_model.item_embeddings.state_dict(), 
                           item_emb_local_path)
                save_remote(item_emb_local_path, item_emb_remote_path, "rb", "wb")
        accelerator.wait_for_everyone()
        
    accelerator.print("-----End Rec GPT Pretraining Loop-----")


if __name__ == "__main__":
    main()