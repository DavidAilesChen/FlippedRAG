import numpy as np
from transformers import AutoModel, AutoTokenizer
# from sentence_transformers import SentenceTransformer, util
from numpy.linalg import norm
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import torch
from torch import nn
from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from sentence_transformers import SentenceTransformer

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(parent_dir)))

class localEmbedding(nn.Module):
    def __init__(self, path:str='', device:str=''):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(path, add_pooling_layer = False, output_hidden_states=False)
        self.embedding.to(device)
        self.pool_type = 'cls'
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # self.tokenizer.to(device)
        self.decive = device
    
    def pooling(self, token_embeddings,input):
        output_vectors = []
        #attention_mask
        attention_mask = input['attention_mask']
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        t = token_embeddings * input_mask_expanded
        
        sum_embeddings = torch.sum(t, 1)
 
        sum_mask = input_mask_expanded.sum(1)
        
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        output_vectors.append(sum_embeddings / sum_mask)
 
        output_vector = torch.cat(output_vectors, 1)
        return  output_vector
    
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        """
        length_sorted_idx = np.argsort([-len(sen) for sen in texts])
        texts = [texts[idx] for idx in length_sorted_idx]
       """
        input_ids = self.tokenizer(texts, max_length=256, padding = "max_length", truncation=True, return_tensors='pt')
        # input_ids.pop('attention_mask')
        input_ids = input_ids.to(self.decive)
        embeddings = self.embedding(**input_ids)
        if self.pool_type == 'mean':
            token_embeddings = embeddings[0]
            embeddings = self.pooling(token_embeddings, input_ids)
        else:
            embeddings = embeddings[0][:, 0, :]
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # embeddings = embeddings.last_hidden_state[:,0,:]
        """
        embeddings = self.embedding.encode(texts)"""
        
        return embeddings.tolist()
    
class localEmbedding_sentence(nn.Module):
    def __init__(self, path:str='', device:str=''):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(grandparent_dir+"/model_hub/msmarco-bert-co-condensor")
        self.embedding.to(device)
        self.pool_type = 'cls'
        self.tokenizer = AutoTokenizer.from_pretrained(grandparent_dir+"/model_hub/msmarco-bert-co-condensor")
        # self.tokenizer.to(device)
        self.decive = device
    
    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:,0]
    
    def encode(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, max_length=256 ,padding=True, truncation=True, return_tensors='pt')
        encoded_input.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output)

        return embeddings
    
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts=texts)
        
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # embeddings = embeddings.last_hidden_state[:,0,:]
        """
        embeddings = self.embedding.encode(texts)"""
        
        return embeddings.tolist()
    
class localEmbedding_contriever(nn.Module):
    def __init__(self, path:str='', device:str=''):
        super().__init__()
        self.embedding = AutoModel.from_pretrained(grandparent_dir+"/model_hub/contriever_msmarco")
        self.embedding.to(device)
        self.pool_type = 'mean'
        self.tokenizer = AutoTokenizer.from_pretrained(grandparent_dir+"/model_hub/contriever_msmarco")
        # self.tokenizer.to(device)
        self.decive = device
    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Tokenize sentences
        # encoded_input = self.tokenizer(texts, max_length=256 ,padding=True, truncation=True, return_tensors='pt')
        encoded_input = self.tokenizer(texts ,padding=True, truncation=True, return_tensors='pt')
        encoded_input.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.mean_pooling(model_output[0], encoded_input['attention_mask'])

        return embeddings
    
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts=texts)
        
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # embeddings = embeddings.last_hidden_state[:,0,:]
        """
        embeddings = self.embedding.encode(texts)"""
        
        return embeddings.tolist()
    
class localEmbedding_ance(nn.Module):
    def __init__(self, path:str='', device:str=''):
        super().__init__()
        self.embedding = SentenceTransformer(grandparent_dir+'/model_hub/msmarco-roberta-base-ance-firstp')
        self.embedding.to(device)
        self.pool_type = 'cls'
        self.tokenizer = AutoTokenizer.from_pretrained(grandparent_dir+"/model_hub/msmarco-roberta-base-ance-firstp")
        # self.tokenizer.to(device)
        self.decive = device
    
    def cls_pooling(self ,model_output, attention_mask):
        return model_output.last_hidden_state[:,0]
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embedding.encode(texts)

        return embeddings
    
    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts=texts)
        
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # embeddings = embeddings.last_hidden_state[:,0,:]
        """
        embeddings = self.embedding.encode(texts)"""
        
        return embeddings.tolist()

class localEmbedding_dpr(nn.Module):
    def __init__(self, path: str = '', device: str = '', language_code: str = 'en_XX'):
        super().__init__()
        self.embedding = SentenceTransformer(grandparent_dir+'/model_hub/dpr')
        self.embedding.to(device)
        self.device = device
        self.language_code = language_code
        # Activate the language-specific adapters
        self.embedding[0].auto_model.set_default_language(language_code)
        
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.embedding.encode(texts, normalize_embeddings=True)
        return embeddings

    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings using the DPR model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.encode(texts=texts)
        return embeddings.tolist()
    
class localEmbedding_QWEN3(nn.Module):
    def __init__(self, path: str = '', device: str = '', language_code: str = 'en_XX'):
        super().__init__()
        self.embedding = SentenceTransformer(path)#'Qwen3-Embedding-0.6B'
        print(path," Loaded!")
        print(self.embedding)
        self.embedding.to(device)
        self.device = device
        self.language_code = language_code
        # Activate the language-specific adapters
        # self.embedding[0].auto_model.set_default_language(language_code)
        
    def encode(self, texts, prompt_name = None):
        if isinstance(texts, str):
            texts = [texts]
        if prompt_name is not None:
            embeddings = self.embedding.encode(texts, prompt_name=prompt_name)
        else:
            embeddings = self.embedding.encode(texts)
            # embeddings = self.embedding.encode(texts, normalize_embeddings=True)
        return embeddings

    def forward(self, text):
        embeddings = self.embed_documents([text])[0]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings using the DPR model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [text.replace("\n", " ") for text in texts]
        embeddings = self.encode(texts=texts)
        return embeddings.tolist()

