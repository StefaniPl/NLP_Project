from transformers import AutoTokenizer
from transformers import AutoModel
import pandas as pd
import warnings
from numba import jit, cuda
warnings.filterwarnings("ignore")
@jit  
def tokenize(model_name, text_batch):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer(text_batch,
                       truncation=True, 
                       max_length=128,
                       return_tensors='pt', 
                       padding=True)
    return tokens
@jit  
def get_representation(model_name, text_batch):
    model = AutoModel.from_pretrained(model_name)
    tokens = tokenize(model_name, text_batch)
    
    return model(input_ids=tokens.input_ids,
                 attention_mask=tokens.attention_mask,
                 output_attentions=False, 
                 output_hidden_states=False)