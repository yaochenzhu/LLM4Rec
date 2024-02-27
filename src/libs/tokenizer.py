'''
MIT License
Copyright (c) 2024 Yaochen Zhu
'''

import re
import numpy as np
from transformers import GPT2Tokenizer

class TokenizerWithUserItemIDTokens(GPT2Tokenizer):
    def __init__(self, 
                 vocab_file, 
                 merges_file, 
                 num_users,
                 num_items,
                 **kwargs):
        super().__init__(vocab_file=vocab_file, 
                         merges_file=merges_file, 
                         **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.user_token_encoder = self._add_user_token_encoder()
        self.item_token_encoder = self._add_item_token_encoder()
        
        #We add the user/item token encoders to the original vocab encoder
        self.encoder.update(self.user_token_encoder)
        self.encoder.update(self.item_token_encoder)
        
        #We add the corresponding decoders to the original vocab decoder
        self.user_token_decoder = {v:k for k,v in self.user_token_encoder.items()}
        self.item_token_decoder = {v:k for k,v in self.item_token_encoder.items()}
        self.decoder.update(self.user_token_decoder)
        self.decoder.update(self.item_token_decoder)
        
    
    def _add_user_token_encoder(self):
        return {"user_{}".format(i):(i+self.vocab_size) 
                for i in range(self.num_users)}
    
    def _add_item_token_encoder(self):
        return {"item_{}".format(j):(j+self.vocab_size+self.num_users)
                for j in range(self.num_items)}
    
    def _pre_tokenize(self, text):
        '''
            In this function, we break down the sentence that 
            describes user/item features or their historical 
            interactions into pieces, where the ID word like
            user_i or item_j is kept as a single piece. 
            
            E.g.,
                text = "This is user_1's comment about item_3 
                        after he bought the item"
                pieces = ['This is', 'user_1', "'s comment about", 
                          'item_3', ' after he bought the item']
                          
            Note that we keep the space on the left of a word to 
            show that the word does not appear on the beginning 
            part of a sentence.
        '''
        pattern = r'(user_\d+|item_\d+)'
        matches = re.findall(pattern, text)
        pieces = re.split(pattern, text)
        pieces = [piece.rstrip() for piece in pieces if piece.rstrip()]
        return pieces
    
    def _tokenize(self, text):
        '''
            Please note that when the token is a user/item token,
            we don't distinguish whether it appears on the start
            of the a sentence or not.
        '''
        split_tokens = []
        pieces = self._pre_tokenize(text)
        for piece in pieces:
            # If piece is a user ID
            # piece is itself a token
            if piece in self.user_token_encoder.keys():
                split_tokens.append(piece)
            # If piece is an item ID
            # piece is also a token
            elif piece in self.item_token_encoder.keys():
                split_tokens.append(piece)
            # If piece is a sentence
            # Use the original tokenization to
            # further break down piece
            else:
                split_tokens += super()._tokenize(piece)
        return split_tokens


class TokenizerWithUserItemIDTokensBatch(TokenizerWithUserItemIDTokens):
    """
     tokenizer class that extends TokenizerWithUserItemIDTokens
     and supports batch encoding.
    """
    def __init__(self, vocab_file, merges_file, num_users, num_items, **kwargs):
        super().__init__(vocab_file=vocab_file, merges_file=merges_file,
                         num_users=num_users, num_items=num_items, **kwargs)
        # Set the padding token ID to 0
        self.pad_token_id = 0
    
    def encode_batch(self, texts, max_length=None):
        """
        Encodes a batch of texts into input IDs and attention masks.

        Args:
            texts (List[str]): List of input texts to be encoded.
            max_length (int, optional): Maximum length of the encoded 
                sequences. If None, the maximum length in the batch 
                will be used. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the 
                input IDs and attention masks as NumPy arrays.
        """
        encoded_inputs = []
        max_length_batch = max(len(self._tokenize(text)) for text in texts)
        
        # Determine the maximum length for padding
        if (not max_length) or max_length <= max_length_batch:
            max_length = max_length_batch
        
        for text in texts:
            tokens = self._tokenize(text)
            input_ids = self.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            
            # Pad the sequence to the max_length
            padding_length = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            
            encoded_inputs.append((input_ids, attention_mask))
        
        input_ids_batch, attention_mask_batch = zip(*encoded_inputs)
        return np.array(input_ids_batch), np.array(attention_mask_batch)
