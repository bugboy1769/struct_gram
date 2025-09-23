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
from pathlib import Path

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

model_dict = {
    "m1": "gpt2",
    "m2": "HuggingFaceTB/SmolLM-135M",
    "m3": "meta-llama/Llama-3.1-8B",
    "m4": "google-t5/t5-small",
    "m5": "meta-llama/Llama-3.2-3B",
}

llm=None

class HFModel():
    def __init__(self):
        self.model=None
        self.tokenizer=None
        self.device=None
    
    def set_model(self, model_key="m1"):
        self.model=AutoModelForCausalLM.from_pretrained(model_dict[model_key])
        self.tokenizer=AutoTokenizer.from_pretrained(model_dict[model_key])
        self.tokenizer.pad_token=self.tokenizer.eos_token
    
    def set_and_move_device(self):
        self.device=torch.device("cuda")
        self.model=self.model.to(self.device)

class vLLMModel():
    def __init__(self):
        self.llm=None
        self.sampling_params=SamplingParams(
            temperature=0.2,
            top_p=0.7,
            top_k=50,
            max_tokens=50
            )

    def load_llm(self):
        self.llm=LLM(
            model="meta-llama/Llama-3.2-3B",
            gpu_memory_utilization=0.8,
           )
    
    def generate_response(self, prompt):
        if self.llm==None:
            raise ValueError("LLM not loaded, call load_llm first.")
        return self.llm.generate(prompt, self.sampling_params)

class DataProcessor:
    def __init__(self):
        self.config={}
        self.supported_formats=[".csv", ".xlsx", ".json", ".parquet"]
        self.data_cache={}
    
    def load_data(self, filename, **kwargs):
        file_ext=Path(filename).suffix.lower()
        if filename in self.data_cache:
            return self.data_cache[filename]
        loaders={
            '.csv':pd.read_csv,
            '.xlsx':pd.read_excel,
            '.json':pd.read_json,
            '.parquet':pd.read_parquet
        }
        if file_ext not in loaders:
            raise ValueError(f"Invalid File Type: {file_ext}")
        df=loaders[file_ext](filename, **kwargs)
        self.data_cache[filename]=df
        return df
    
    def validate_data(self, df):
        report={
            'errors':[],
            'warnings':[],
            "info": {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'dtypes': df.dtypes.to_dict()
            }
        }
        if df.empty:
            report["errors"].append("DataFrame is Empty.")
        if len(df.columns) < 2:
            report["errors"].append("DataFrame has only 1 column, relationality cannot be established.")
        if df.memory_usage(deep=True).sum()>1e9:
            report["warnings"].append("DataFrame size is exceeding 1GiB, may cause memory issues.")
        return report

    def truncate_data(self, df, max_rows=None, strategy='head'):
        if max_rows is None:
            max_rows=self.config.get('max_rows', 15)
        if len(df)<=max_rows:
            return df
        strategies={
            'head': lambda:df.head(max_rows),
            'tail': lambda:df.tail(max_rows),
            'sample': lambda:df.sample(max_rows, random_state=42)
        }
        return strategies[strategy]()

    def get_data_summary(self, df):
        return {
            'shape':df.shape,
            'columns':list(df.columns),
            'dtypes':df.dtypes.to_dict(),
            'null_counts':df.isnull().sum().to_dict(),
            'numeric_columns':df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns':df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum()/1024**2
        }

    def preprocess_columns(self, df): #Think about impact on training, should we include messy column names?
        df_clean=df.copy()
        df_clean.columns=df_clean.columns.str.strip()
        df_clean.columns=df_clean.columns.str.replace(' ', '_')
        return df_clean

    def handle_missing_data(self, df, strategy='report'):
        if strategy=='report':
            return df.isnull().sum().to_dict()
        if strategy=='drop':
            return df.dropna()
        if strategy=='fill':
            return df.fillna(df.mean(numeric_only=True))
        else:
            return df

    def detect_column_types(self, df):
        column_types={}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique()<10:
                    column_types[col]='categorical_numeric'
                else:
                    column_types[col]='continuous_nmeric'
            elif pd.api.types.is_object_dtype(df[col]):
                if df[col].str.match(r'\d{4}-\d{2}-\d{2}').any():
                    column_types[col]='date_string'
                else:
                    column_types[col]='text'
        return column_types

class ColumnStatsExtractor:
    def __init__(self):
        self.digit_pattern=r'\d'
        self.special_char_pattern=r'[^a-zA-Z0-9\s]'
    def get_col_stats(self, df, col):
        series = df[col]
        stats = {f"column_name {col}":{
            "column_name": f" {col}",
            "column_data_type": f" {str(series.dtype)}",
            "unique_count": f" {len(series.unique())}",
            "null_count": f" {series.isnull().sum()}"
        }}
        if pd.api.types.is_numeric_dtype(series):
            stats[f"column_name {col}"].update(self.extract_numeric_stats(series))
        if pd.api.types.is_string_dtype(series):
            stats[f"column_name {col}"].update(self.extract_string_stats(series))
        return stats
    def extract_numeric_stats(self, series):
        return {
                "mean": f" {series.mean()}",
                "standard_deviation": f" {series.std()}",
            }
    def extract_string_stats(self, series):
        return {
                "avg_length_elements": f" {series.str.len().mean()}",                "contains_numbers": f" {series.str.contains(self.digit_pattern).any()}",
            }
    #ToDo: Add datetime and categorical stat extractors
    def get_batch_stats(self, df, columns=None):
        if columns is None:
            columns=df.columns.to_list()
        batch_stats={}
        for col in columns:
            batch_stats[col]=self.get_col_stats(df, col)
        return batch_stats

class RelationshipGenerator:
    

def generate_vllm(prompt):
    return llm.generate(prompt, vllm_llm.sampling_params)


def table_to_nx_graph(df):
    G=nx.Graph()
    node_features=[]
    max_len=0
    for col in df.columns:
        col_stats=get_col_stats(df, col)
        G.add_node(f"col_{col}", node_type="column", col_stats)



