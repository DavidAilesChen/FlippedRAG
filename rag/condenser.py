import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from torch import nn
import numpy as np
import faiss
from transformers import BertModel, AutoModel, XLMRobertaForMaskedLM, AutoModelForCausalLM, AutoModelForMaskedLM
from transformers import AutoTokenizer

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"
# 
# tokenizer = AutoTokenizer.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")


def condenser_example(text):
    model = AutoModel.from_pretrained('Luyu/co-condenser-marco')
    print(model)
    text_ids = tokenizer.encode(text)
    text_ids = torch.tensor([text_ids])
    print(text_ids)
    output = model(text_ids)
    print(output[0].shape, output[1].shape)

def pooling(token_embeddings,input):
    output_vectors = []
    #attention_mask
    attention_mask = input['attention_mask']
    #[B,L]------>[B,L,1]------>[B,L,768]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    t = token_embeddings * input_mask_expanded
    #[B,768]
    sum_embeddings = torch.sum(t, 1)
 
    # [B,768]
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    output_vectors.append(sum_embeddings / sum_mask)
 
    output_vector = torch.cat(output_vectors, 1)
 
    return  output_vector


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
        if isinstance(score, torch.Tensor):
            score = score.cpu().detach().numpy()
        similarity = np.diag(score, k=0)
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
    
    query_emb = model(query_ids)[0][:, 0, :]
    passage_emb = model(passage_ids)[0][:, 0, :]
    """
    query_emb = model.embeddings(query_ids)
    passage_emb = model.embeddings(passage_ids)
    print(query_emb.shape)
    query_encode = model.encoder(query_emb)
    passage_encode = model.encoder(passage_emb)"""
    similarity = compute_similarity(query_emb, passage_emb)
    similarity = torch.tensor(similarity).to(device)
    return similarity, query_emb, passage_emb

def condenser_loss(query_emb, passage_emb, labels, device, args):
    similarity = compute_similarity(query_emb, passage_emb)
    similarity = torch.tensor(similarity).to(device)
    loss_function = nn.MSELoss()
    loss = loss_function(similarity, labels)
    loss.requires_grad_(True)
    return loss

def sim_score(model, query, passage, measure = 'dot'):
    query_emb = model.encode(query)
    passage_emb = model.encode(passage)
    if type(query_emb) != np.ndarray:
        query_emb = query_emb.cpu().numpy()
        passage_emb = passage_emb.cpu().numpy()
    d = query_emb.shape[1]
    if measure == 'L2':
        index = faiss.IndexFlatL2(d)
    elif measure == 'dot':
        index = faiss.IndexFlatIP(d)
    index.add(np.array(passage_emb))
    dis, ind = index.search(np.array(query_emb), 1)
    distance = dis[0][0]
    return distance

def sim_score_for_passage_list(model, query, candidates, k=10, measure = "dot"):
    if not isinstance(candidates, list):
        query_emb = model.encode_(query)[0].cpu().numpy()
        doc_embs = model.encode_(candidates).cpu().numpy()
    else:
        try:
            # query_emb = model.encode([query], prompt_name="query")[0]
            query_emb = model.encode([query])[0]
            # print("no query!")
            doc_embs = model.encode(candidates)
        except:
            query_emb = model.encode([query])[0]
            doc_embs = model.encode(candidates)
        if type(query_emb) != np.ndarray:
            query_emb = query_emb.cpu().numpy()
            doc_embs = doc_embs.cpu().numpy()
    
    d = len(query_emb)
    query_emb = np.array([query_emb])

    if measure == 'L2':
        index = faiss.IndexFlatL2(d)
    elif measure == 'dot':
        index = faiss.IndexFlatIP(d) 
    index.add(doc_embs) 
    dis, ind = index.search(query_emb, k)
    dis_score = [0]*len(dis[0])
    for i in range(len(dis_score)):
         dis_score[ind[0][i]] = dis[0][i]
    
    return dis_score, ind[0]

def sim_ranker(model, query, candidates, k=10, measure = "dot"):
    """
    INPUT:
    dense model;str;list
    """
    if not isinstance(candidates, list):
        query_emb = model.encode_(query).cpu().numpy()
        doc_embs = model.encode_(candidates).cpu().numpy()
    else:
        query_emb = model.encode(query).cpu().numpy()
        doc_embs = model.encode(candidates).cpu().numpy()

    d = query_emb.shape[1]
    dis_score = []
    for i in range(query_emb.shape[0]):
        if measure == 'L2':
            index = faiss.IndexFlatL2(d)
        elif measure == 'dot':
            index = faiss.IndexFlatIP(d)
        # index = faiss.IndexFlatIP(d)

        index.add(np.array([doc_embs[i]])) 
        dis, ind = index.search(np.array([query_emb[i]]), 1)
        dis_score.append(dis[0][0])
    
    # for i in range(len(dis_score)):
    #      dis_score[ind[0][i]] = dis[0][i]
    
    return [dis_score], ind


if __name__ == '__main__':
    condenser_example("We subsequently investigate whether there is a relationship between personality traits and ChatGPT's political biases.")
