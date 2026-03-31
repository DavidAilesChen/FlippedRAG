#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
curdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(curdir))
from typing import List, Dict, Tuple, Set, Any, Union
from collections import Counter
import re
# import jieba
# import jieba.analyse
import numpy as np
import spacy
from rag.rag_utils import extract_by_symbol

def extract_keywords(text: str, top_k: int = 50,pos_tags=["NOUN", "PROPN", "ADJ", "ADV", "NUM", "SYM", "X"]) -> List[str]:#, "INTJ"
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # keywords = jieba.analyse.textrank(text, topK=top_k)
    words = [
        token.text for token in doc
        if token.pos_ in pos_tags 
        and not token.is_stop 
        and not token.is_punct
    ]
    word_freq = Counter(words)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    keywords = [word for word, freq in sorted_words[:top_k]]
    
    phrases = []
    current_phrase = []
    for token in doc:
        if token.pos_ in pos_tags:
            current_phrase.append(token.text)
        else:
            if len(current_phrase) >= 2:
                phrases.append(" ".join(current_phrase))
            current_phrase = []
    if len(current_phrase) >= 2:
        phrases.append(" ".join(current_phrase))
    key_count = Counter(phrases)
    key_count.update(word_freq)
    
    result = keywords + phrases[:min(top_k, len(phrases))]
    
    return result, key_count


def secure_keyword_aggregation(
    query: str, 
    llm: Any, 
    p_list: List[str], 
    alpha: float = 0.1, 
    beta: int = 3
) -> str:
    counter = Counter()
    n = 0
    
    for passage in p_list:
        prompt = f"""Answer the query given retrieved passages. Say 'I don't know' if no relevant information found. Put the answer in [[ ]].
Query: {query}
Retrieved passage: {passage}
Answer:"""
        
        llm_output = llm(prompt)
        response = extract_by_symbol(llm_output, symbol = "[[ ]]")[0]  
        print("single_response: ", response)
        
        if "I don't know" not in response:
            n += 1 
            
            keywords, keyword_count= extract_keywords(response)

            counter.update(keyword_count)
        else:
            print("Existing I don't know!")
    
    threshold = min(alpha * n, beta)
    print("CCCCounter: ", counter)

    filtered_keywords = [
        keyword for keyword, count in counter.items() 
        if count >= threshold
    ]
    print("$$$$Filtered keys:", filtered_keywords)

    filtered_keywords.sort()
    
    if not filtered_keywords:
        # return "I'm sorry, but I don't have relevant information to answer your query."
        return "This topic is debatable and does not have a correct answer."
    
    final_prompt = f"""Answer the query using provided keywords. Put the answer in [[ ]].
Query: {query}
Keywords: {', '.join(filtered_keywords)}
Answer:"""

    final_answer = llm(final_prompt)
    print("LLM_output: ", final_answer)
    output_answer = extract_by_symbol(final_answer, symbol = "[[ ]]")
    if len(output_answer[0].strip().replace(" ", "")) == 0 or True:
        return final_answer
    else:
        return ' '.join(output_answer)


def secure_decoding_aggregation(
    query: str,
    llm: Any,
    p_list: List[str],
    alpha: float = 0.3,
    prob: float = 0.001,
    tmax: int = 256
) -> str:
    r_list = []
    r = ""
    from tqdm import tqdm
    
    relevant_passages = []
    for passage in p_list:
        prompt = f"""Answer the query given retrieved passages. Say 'I don't know' if no relevant information found. Put the answer in [[ ]]
        
Query: {query}
Retrieved passage: {passage}
"""
        
        response = extract_by_symbol(llm(prompt), symbol = "[[ ]]")[0]
        print("response:", response)
        if "I don't know" not in response:
            relevant_passages.append(passage)
    
    for t in tqdm(range(1, tmax + 1)):
        probability_vectors = []
        for passage in relevant_passages:
            prompt = f"""Answer the query given retrieved passages. Say 'I don't know' if no relevant information found. 
            
Query: {query}
Passage: {passage}
Current text: {r}"""
            
            vector, _ = llm.get_next_token_probabilities(prompt, 5)
            probability_vectors.append(np.array(vector.cpu()))
        
        if probability_vectors:
            probability_vectors = np.array(probability_vectors)
            V = np.mean(probability_vectors, axis=0)
            
            top_two_indices = np.argsort(V)[-2:]
            top_two_probs = V[top_two_indices]
            
            if top_two_probs[1] - top_two_probs[0] > prob or True:
                r_list.append(top_two_indices[1])
                next_token = llm.index_to_token(top_two_indices[1])

            else:
                print("No passages!!", r)
                prompt = f"""Given the query and current text, predict the next token in [[ ]].
                
                        Query: {query}
                        Current text: {r}"""
                tempt_res = llm(prompt)
                print("Tempt_res: ", tempt_res)
                next_token = extract_by_symbol(tempt_res, symbol = "[[ ]]")[0]
        else:

            prompt = f"""Given the query and current text, predict the next token in [[ ]].
            
                Query: {query}
                Current text: {r}"""
            next_token = extract_by_symbol(llm(prompt), symbol = "[[ ]]")[0]
        
        # r_list.append()
        # r = llm.tokenizer.convert_tokens_to_string(r_list)
        r = llm.tokenizer.decode(r_list)
        # r += next_token
        
        if next_token in ["。", "\n\n", "!", "！", "?", "？", "</s>"]:
            break
    
    print("R:", r)
    
    return "[["+r+"]]"



if __name__ == "__main__":
    test_query = "What is Retrieval Augmented Generation?"
    test_passages = [
        "Retrieval Augmented Generation (RAG) expands the capabilities of modern large language models (LLMs), by anchoring, adapting, and personalizing their responses to the most relevant knowledge sources. It is particularly useful in chatbot applications, allowing developers to customize LLM output without expensive retraining. Despite their significant utility in various applications, RAG systems present new security risks. In this work, we propose new attack vectors that allow an adversary to inject a single malicious document into a RAG system's knowledge base, and mount a backdoor poisoning attack. We design Phantom, a general two-stage optimization framework against RAG systems, that crafts a malicious poisoned document leading to an integrity violation in the model's output. First, the document is constructed to be retrieved only when a specific trigger sequence of tokens appears in the victim's queries. Second, the document is further optimized with crafted adversarial text that induces various adversarial objectives on the LLM output, including refusal to answer, reputation damage, privacy violations, and harmful behaviors.",
        "In this paper, we comprehensively study the risk of knowledge base poisoning to maliciously influence RAG systems. We define a novel threat model of backdoor poisoning in RAG systems, in which an adversary crafts a single malicious file embedded in the RAG knowledge base to achieve an integrity violation when a natural trigger appears in user queries. We introduce Phantom, a general two-stage optimization framework that induces a range of adversarial objectives in the LLM generator, as shown in Figure 1, by poisoning a single document.",
    ]
    
    model_path_llama3 = '/Meta-Llama-3-8B-Instruct'
    model_path_mixtral = "/Mixtral-8x7B-Instruct-v0.1"
    model_path_vicuna = "/vicuna/vicuna-13b-v1.5"
    from rag.llama3 import LLaMA3_LLM
    from rag.Mixtral import Mixtral_LLM
    from rag.Vicuna import Vicuna_LLM
    # llm = LLaMA3_LLM(mode_name_or_path=model_path_llama3)
    # llm = Mixtral_LLM(model_name=model_path_mixtral)
    llm = Vicuna_LLM(model_name = model_path_vicuna)
    llm.eval()
    # print(extract_keywords(test_passages[0]))
    # result = secure_keyword_aggregation(test_query, llm, test_passages)
    result = secure_decoding_aggregation(test_query, llm, test_passages)
    