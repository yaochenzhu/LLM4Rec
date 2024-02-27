'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import GPT2Model, GPT2Config


class GPT4RecommendationBaseModel(nn.Module):
    '''
        The base class for collaborative GPT model, i.e.,
        the GPT model with extra user/item embeddings
    '''
    def __init__(self, config, gpt2model):
        super(GPT4RecommendationBaseModel, self).__init__()
        # Obtain the number of users, items
        # and the size of the original vocabulary
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size
        self.config = config

        # Create new token embeddings for user/item tokens
        self.user_embeddings = nn.Embedding(self.num_users, config.n_embd)
        self.item_embeddings = nn.Embedding(self.num_items, config.n_embd)

        # Randomly initialize the new token embeddings
        self.user_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.item_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)
        
        # The pretrained gpt2 model
        self.gpt2model = gpt2model
        
    def embed(self, input_ids):
        # input_ids is a tensor of shape (batch_size, seq_length)
        vocab_mask = (input_ids < self.vocab_size).long() 
        user_mask = ((input_ids >= self.vocab_size) & (input_ids < self.vocab_size + self.num_users)).long() 
        item_mask = (input_ids >= self.vocab_size + self.num_users).long()
        
        # IDs outside of vocab range are set to 0
        vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size-1)  
        vocab_embeddings = self.gpt2model.wte(vocab_ids)
        vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)
        
        # IDs outside of user range are set to 0
        user_ids = ((input_ids - self.vocab_size) * user_mask).clamp_(0, self.num_users-1)
        user_embeddings = self.user_embeddings(user_ids)
        user_embeddings = user_embeddings * user_mask.unsqueeze(-1)
        
        # IDs outside of item range are set to 0
        item_ids = ((input_ids - self.vocab_size - self.num_users) * item_mask).clamp_(0, self.num_items-1)
        item_embeddings = self.item_embeddings(item_ids)
        item_embeddings = item_embeddings * item_mask.unsqueeze(-1)

        # Sum up the embeddings as the input embeddings
        input_embeddings = vocab_embeddings + user_embeddings + item_embeddings
        return input_embeddings
        
    def forward(self, input_ids=None, **kwargs):
        # Obtain the embeddings of the input id sequence
        input_embeddings = self.embed(input_ids)
        # The input_embeds will be summed up with the pos_embed
        # And then forward into the transformer to get the results
        return self.gpt2model(inputs_embeds=input_embeddings, **kwargs)


class CollaborativeGPTwithItemLMHeadBatch(nn.Module):
    '''
        Collaborative filtering model to learn user/item embeddings.
    '''
    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemLMHeadBatch, self).__init__()

        # Obtain the number of users, items, and vocabulary size
        self.num_users = config.num_users
        self.num_items = config.num_items
        self.vocab_size = config.vocab_size

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Item recommendation head
        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)
        
        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight 

    def forward(self,
                input_ids_prompt,
                input_ids_main,
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                content_embeds=None,
                **kwargs):
        # Base model forward pass for the prompt text
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, 
                                         **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Base model forward pass for the main text with attention mask
        outputs_main = self.base_model(input_ids=input_ids_main,
                                       past_key_values=past_key_values,
                                       attention_mask=attention_mask,
                                       return_dict=True)

        item_logits = self.item_head(outputs_main.last_hidden_state)
        outputs = (item_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = item_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            shift_labels = shift_labels - self.vocab_size - self.num_users

            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
        
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the item sequences
            loss = loss_fct(active_logits, active_labels)
            
            # Mutual regularization loss
            if regularize:
                collaborative_embeds =  torch.cat(
                    (self.base_model.embed(input_ids_prompt),
                     self.base_model.embed(input_ids_main)),
                    axis=1
                )
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        collaborative_embeds,
                        content_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs
            else:
                outputs = (loss,) + outputs
        return outputs


class ContentGPTForUserItemWithLMHeadBatch(nn.Module):
    '''
        This class conducts language modeling to learn both
        user/item token embeddings via textual data, where
        we view the texts that include user/item ID as prompt.
        E.g.,
            inputs_ids_prompt:
              "user_1 writes the following review for item_1:"
            inputs_ids_main:
              "This item is too expensive."
        where we only calculate LM loss on the main texts.
    '''
    def __init__(self, config, base_model):
        super(ContentGPTForUserItemWithLMHeadBatch, self).__init__()
        self.base_model = base_model
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between the output layer and the token embeddings
        self.lm_head.weight = self.base_model.gpt2model.wte.weight

    def forward(self, 
                input_ids_prompt, 
                input_ids_main, 
                labels_main=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                collaborative_embeds=None,
                **kwargs):
        outputs_prompt = self.base_model(input_ids=input_ids_prompt, 
                                         return_dict=True, **kwargs)
        past_key_values = outputs_prompt.past_key_values

        # Calculate the language modeling loss for the main texts
        outputs_main = self.base_model(input_ids=input_ids_main, 
                                       past_key_values=past_key_values, 
                                       attention_mask=attention_mask,
                                       return_dict=True)

        lm_logits = self.lm_head(outputs_main.last_hidden_state)
        outputs = (lm_logits,) + outputs_main[1:]

        if labels_main is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()
            
            # Define the loss function
            loss_fct = CrossEntropyLoss()

            # Calculate the loss only where attention mask is one
            prompt_length = input_ids_prompt.shape[1]
            main_length = input_ids_main.shape[1]
            
            active_loss = attention_mask[:, prompt_length+1:].reshape(-1) == 1
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
            active_labels = shift_labels.view(-1)[active_loss]

            # Language modeling loss for the token sequences
            loss = loss_fct(active_logits, active_labels)
            
            # Mutual regularization loss
            if regularize:
                # User/Item token embeddings only appear in the prompt
                content_embeds = self.base_model.embed(input_ids_prompt)
                regularize_loss = lambda_V * torch.mean(
                    nn.MSELoss(reduction='sum')(
                        content_embeds,
                        collaborative_embeds)
                )
                loss += regularize_loss
                outputs = (loss, regularize_loss) + outputs            
            else:
                outputs = (loss,) + outputs
        return outputs


class CollaborativeGPTwithItemRecommendHead(nn.Module):
    '''
        Recommend items to a user according to input queries.
        multinomial likelihood is put on all the items for a user.
    '''
    def __init__(self, config, base_model):
        super(CollaborativeGPTwithItemRecommendHead, self).__init__()
        # Obtain the number of users and items
        self.num_users = config.num_users
        self.num_items = config.num_items

        # Base GPT model with extended user/item ID token embeddings
        self.base_model = base_model
        
        # Item recommendation head
        self.item_head = nn.Linear(config.n_embd, self.num_items, bias=False)
        
        # Tie the weights between the item embeddings and the item recommendation head
        self.item_head.weight = self.base_model.item_embeddings.weight 

    def forward(self, 
                input_ids=None, 
                target_ids=None,
                attention_mask=None,
                regularize=False,
                lambda_V=None,
                main_ids=None,
                content_embeds=None,
                **kwargs):    
        transformer_outputs = self.base_model(input_ids, 
                                              attention_mask=attention_mask, 
                                              **kwargs)
        hidden_states = transformer_outputs[0]

        # Find the indices of the last non-padding tokens
        last_non_pad_token_indices = attention_mask.sum(dim=1) - 1

        # Gather the last non-padding token embeddings
        last_token_hidden_states = torch.stack([
            hidden_states[i, idx, :] for i, idx in \
                enumerate(last_non_pad_token_indices)
        ])

        # Calculate the item scores
        item_scores = self.item_head(last_token_hidden_states)

        # Convert scores to multinomial probabilities
        item_log_probs = F.log_softmax(item_scores, dim=-1)
        
        # Calculating the multinomial loss
        neg_ll = -torch.mean(torch.sum(item_log_probs * target_ids, dim=-1))
        
        if regularize:
            # User/Item token embeddings only appear in the prompt
            rec_embeds_prompt = self.base_model.embed(input_ids)
            rec_embeds_target = self.base_model.embed(main_ids)
            rec_embeds = torch.cat(
                (rec_embeds_prompt, rec_embeds_target),
                axis=1
            )
            regularize_loss = lambda_V * torch.mean(
                nn.MSELoss(reduction='sum')(
                    rec_embeds,
                    content_embeds)
            )
            neg_ll += regularize_loss
            outputs = (neg_ll, regularize_loss, item_log_probs)
        else: 
            outputs = (neg_ll, item_log_probs)
        return outputs
