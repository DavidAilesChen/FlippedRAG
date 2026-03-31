from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch

class Vicuna_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name, device="cuda"):#vicuna-13b-v1.5
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Loading:",model_name,"...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    # def generate_(self, prompt, max_new_tokens=512, temperature=0.7, stop: Optional[List[str]] = None, 
    #             run_manager: Optional[CallbackManagerForLLMRun] = None,
    #             **kwargs: Any):
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    #     with torch.no_grad(): 
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_new_tokens=max_new_tokens,
    #             temperature=temperature,
    #             do_sample=True,
    #         )
    #     response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    #     return response
    
    def _call(self, prompt: str,  stop: Optional[List[str]] = None, max_new_tokens=512, temperature=0.7, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad(): 
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
        return response
    
    def get_next_token_probabilities(self, prompt : str, topk : int, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():    
            outputs = self.model(
                **inputs, 
                )
        logits = outputs.logits[0, -1]  
        probs = torch.softmax(logits, dim=-1)
        token_probs = {
            self.tokenizer.decode([idx]): float(probs[idx])
            for idx in torch.topk(probs, k=topk).indices 
        }
        return probs, token_probs

    def index_to_token(self, token_ids):
        if isinstance(token_ids, int):
                return self.tokenizer.decode([token_ids])
        return self.tokenizer.decode(token_ids) 
    
    @property
    def _llm_type(self) -> str:
        return "vicuna"
    
    def eval(self):
        self.model = self.model.eval()