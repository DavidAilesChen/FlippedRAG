from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch

class LLaMA3_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
        
    def __init__(self, mode_name_or_path :str):

        super().__init__()
        print("Loading...")
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, device_map="auto").eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Finish loading")

    def bulid_input(self, prompt, history=[]):
        user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
        assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>'
        # history.append({'role':'user','content':prompt})
        history= [{'role':'user','content':prompt}]
        prompt_str = ''
        for item in history:
            if item['role']=='user':
                prompt_str+=user_format.format(content=item['content'])
            else:
                prompt_str+=assistant_format.format(content=item['content'])
        return prompt_str
    
    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):

        input_str = self.bulid_input(prompt=prompt)
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        with torch.no_grad():    
            outputs = self.model.generate(
                input_ids=input_ids, max_new_tokens=512, do_sample=True,
                top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=self.tokenizer.encode('<|eot_id|>')[0], pad_token_id=self.tokenizer.eos_token_id
                )
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = self.tokenizer.decode(outputs).strip().replace('<|eot_id|>', "").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()
        return response
    
    def get_next_token_probabilities(self, prompt : str, topk : int, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        input_str = self.bulid_input(prompt=prompt)
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').to(self.model.device)
        with torch.no_grad():    
            outputs = self.model(
                input_ids=input_ids,
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
        return "LLaMA3_LLM"
    
    def eval(self):
        self.model = self.model.eval()