import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
import torch
from pro_data import get_msmarco
from condenser import compute_similarity, condenser_encode

def main():
    CON_NAME = '/model_hub/condenser/co-condenser-marco/'

    target_q, passages, texts = get_msmarco()
    print(len(passages))
    print(texts)

    query = target_q[0][1]
    embedding = AutoModel.from_pretrained(CON_NAME, add_pooling_layer = False, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(CON_NAME)

    query_input_ids = tokenizer([query], max_length=128, padding = 'max_length', truncation=True, return_tensors='pt')['input_ids']
    passages_input_ids = tokenizer(texts, max_length=128, padding = 'max_length', truncation=True, return_tensors='pt')['input_ids']
    print(query_input_ids.shape)
    print("PAS EMB:",passages_input_ids.shape)

    similarity, query_emb, passage_emb = condenser_encode(embedding, query_input_ids, passages_input_ids, 'cuda', args=None)
    print("SIM:",similarity)

if __name__ == '__main__':
    main()