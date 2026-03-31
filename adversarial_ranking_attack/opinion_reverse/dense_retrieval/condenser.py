import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch import nn
import numpy as np
from transformers import BertModel, AutoModel, XLMRobertaForMaskedLM, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import AutoTokenizer

parent_dir = os.path.dirname(__file__)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(os.path.dirname(grandparent_dir)))+"/model_hub"

# tokenizer = AutoTokenizer.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")


def condenser_example(text):
    model = AutoModel.from_pretrained('Luyu/co-condenser-marco')
    print(model)
    text_ids = tokenizer.encode(text)
    text_ids = torch.tensor([text_ids])
    print(text_ids)
    output = model(text_ids)
    print(output[0].shape, output[1].shape)

def compute_norm_product(vector1, vector2):  
    assert vector1.dim() == 1 and vector2.dim() == 1, "Both inputs should be 1D tensors."  
      
    norm1 = vector1.norm()  
    norm2 = vector2.norm()  
       
    return norm1 * norm2 

def mat_norm(mat_1, mat_2):
    values = []
    for i in range(mat_1.shape[0]):
        vector1 = mat_1[i]
        vector2 = mat_2[i]
        norm_value = compute_norm_product(vector1=vector1, vector2=vector2)
        values.append(norm_value)
    values = torch.tensor(values)
    return values          

def compute_similarity(q_reps, p_reps):
    if len(p_reps.size()) == 2:
        score = torch.matmul(q_reps, p_reps.transpose(0, 1))
        # if isinstance(score, torch.Tensor):
        #     score = score.cpu().detach().numpy()
        similarity = torch.diag(score)
        # similarity = np.diag(score, k=0)
    else:
        score = torch.matmul(q_reps, p_reps.transpose(-2, -1))
        if isinstance(score, torch.Tensor):
            score = score.cpu().detach().numpy()
        similarity = np.diagonal(score, axis1=1, axis2=2)
    return similarity

def norm_sim(similarity, q_reps, p_reps):
    norm_values = mat_norm(q_reps,p_reps)
    similarity = similarity/norm_values
    return similarity

def condenser_encode(model,  query_encoding, passage_encoding, device ,args):
    try:
        query_ids = query_encoding['input_ids'].to(device)
        passage_ids = passage_encoding['input_ids'].to(device)
    except:
        query_ids = query_encoding
        passage_ids = passage_encoding
    
    query_emb = model.encode_({'input_ids':query_ids})
    passage_emb = model.encode_({'input_ids':query_ids})
    # query_emb = model(query_ids)[0][:, 0, :]
    # passage_emb = model(passage_ids)[0][:, 0, :]
    """
    query_emb = model.embeddings(query_ids)
    passage_emb = model.embeddings(passage_ids)
    print(query_emb.shape)
    query_encode = model.encoder(query_emb)
    passage_encode = model.encoder(passage_emb)
    """
    similarity = compute_similarity(query_emb, passage_emb)
    similarity = torch.tensor(similarity).to(device)
    return similarity, query_emb, passage_emb

def condenser_loss(query_emb, passage_emb, labels, device, args):
    similarity = compute_similarity(query_emb, passage_emb)
    similarity = torch.tensor(similarity).to(device)
    print("S,", similarity)
    loss_function = nn.MSELoss()
    loss = loss_function(similarity, labels)
    loss.requires_grad_(True)
    return loss

if __name__ == '__main__':
    condenser_example("We subsequently investigate whether there is a relationship between personality traits and ChatGPT's political biases.")
