import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
import pandas as pd
import numpy as np
import itertools
from llm_call import generate_batch_without_decode
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from gcn_conv import TableGCN
from projection_layer import LLMProjector
import time
from langchain_ollama import OllamaLLM
from vllm import LLM, SamplingParams

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
    "m3": "meta-llama/Llama-3.1-8B",
    "m4": "google-t5/t5-small",
    "m5": "meta-llama/Llama-3.2-3B",
}

llm=None

class HFModel(AutoModelForCausalLM, AutoTokenizer, SamplingParams):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def set_model(self):
        self.model=AutoModelForCausalLM.from_pretrained(model_dict["m1"])
        self.tokenizer=AutoTokenizer.from_pretrained(model_dict["m1"])
    
    def set_and_move_device(self):
        self.device=torch.device("cuda")
        self.model=self.model.to(self.device)

class vLLMModel(LLM, SamplingParams):
    def __init__(self):
        super().__init__()
    
    self.sampling_params=SamplingParams(
        temperature=0.2,
        top_p=0.7,
        top_k=50,
        max_tokens=100
    )

    def load_llm():
        global llm
        llm=LLM(
            model="meta-llama/Llama-3.2-3B",
            gpu_memory_utilization=0.8,
        )
    
    def generate_response(prompt):
        return llm.generate(prompt, self.sampling_params)

vllm_llm=vLLMModel()

def load_data(filename):
    path = str(filename)
    return pd.read_csv(path)


def generate_vllm(prompt):
    return llm.generate(prompt, vllm_llm.sampling_params)


