#!/usr/bin/env python
# -*- coding: utf-8 -*-


import re
from typing import List, Dict, Tuple, Any
# import jieba
from nltk import word_tokenize 
import nltk

def tokenize_text(text: str) -> List[str]:
    # return list(jieba.cut(text))
    # print("tokenize_text:",text)
    return word_tokenize(text)


def calculate_keyword_density(query: str, document: str) -> float:
    query_tokens = tokenize_text(query)
    doc_tokens = tokenize_text(document)
    
    keyword_count = 0
    for query_token in query_tokens:
        keyword_count += doc_tokens.count(query_token)
    
    if len(doc_tokens) == 0:
        return 0.0
    
    return keyword_count / len(doc_tokens)


def calculate_sliding_window_density(query: str, document: str, window_size: int = 20, step_size: int = 5) -> List[float]:
    query_tokens = tokenize_text(query)
    doc_tokens = tokenize_text(document)
    if len(doc_tokens) <= window_size:
        return [calculate_keyword_density(query, document)]
    
    densities = []
    
    for start_idx in range(0, len(doc_tokens) - window_size + 1, step_size):
        window_tokens = doc_tokens[start_idx:start_idx + window_size]
        
        keyword_count = 0
        for query_token in query_tokens:
            keyword_count += window_tokens.count(query_token)
        
        window_density = keyword_count / len(window_tokens)
        densities.append(window_density)
    
    return densities


def process_documents(query: str, documents: List[str], window_size: int = 50, step_size: int = 25) -> Dict[int, List[float]]:
    result = {}
    from tqdm import tqdm
    
    for doc_id, document in tqdm(enumerate(documents)):
        window_densities = calculate_sliding_window_density(
            query, document, window_size, step_size
        )
        
        result[doc_id] = window_densities
    
    return result


if __name__ == "__main__":
    # for test
    query = "Should Corporal Punishment Be Used in K-12 Schools?"
    documents = [
        "be used in high school, and would allow students to use the schools When given the choice, some students frequently choose corporal punishment because it is a quick punishment that doesn't cause older children to miss class or other activities, or younger children to miss their valued time on the playground......",
        "Some experts state that corporal punishment prevents children from persisting in their bad behavior and growing up to be criminals.",
        "Should Corporal Punishment Be Used in K-12 Schools? Occasional use of corporal punishment for serious behavioral issues is appropriate because time-out or taking away a toy may not work to correct behavior in a particularly willful or rambunctious child. The negative effects of corporal punishment cited by critics are attached to prolonged and excessive use of the punishment."
    ]
    
    window_size = 50
    step_size = 5
    
    result = process_documents(query, documents, window_size, step_size)
    
    for doc_id, densities in result.items():
        print(f"Doc {doc_id}:")
        print(f"  Overall density: {calculate_keyword_density(query, documents[doc_id]):.4f}")
        print(f"  Window density: {[f'{d:.4f}' for d in densities]}")
