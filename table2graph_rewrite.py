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
                "avg_length_elements": f" {series.str.len().mean()}",
                "contains_numbers": f" {series.str.contains(self.digit_pattern).any()}",
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
    def __init__(self, threshold_config=None):
        self.thresholds=threshold_config or {
            'composite_threshold':0.4,
            'weights': {
                'name_similarity':0.2,
                'value_similarity':0.3,
                'jaccard_overlap':0.2,
                'cardinality_similarity':0.15,
                'dtype_similarity':0.15
            }
        }
        self.sample_size=100 #For Jaccard~1000
    def compute_all_relationship_scores(self, df, stats_dict=None):
        relationships=[]
        columns=df.columns.to_list()
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1]:
                edge_features=self.compute_edge_features(df, col1, col2)
            composite_score=self.compute_composite_score(edge_features)
            if composite_score<=self.thresholds[composite_score]:
                relationships.append({
                    'col1':col1, #why col1 and col2?
                    'col2':col2,
                    'egde_features':edge_features,
                    'composite_score':composite_score
                })
        return relationships
    def compute_edge_features(self, df, col1, col2):
        return {
            'name_similarity':self.cosine_similarity_names(col1, col2),
            'value_similarity':self.cosine_similarity_values(df[col1], df[col2]),
            'jaccard_overlap':self.sampled_jaccard_overlap(df[col1], df[col2]),
            'cardinality_similarity':self.cardinality_similarity(df[col1], df[col2]),
            'dtype_similarity':self.dtype_similarity(df[col1], df[col2])
        }
    
    def cosine_similarity_names(self, col1, col2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        col1_tokens=' '.join(col1.lower().replace('_', ' ').split())
        col2_tokens=' '.join(col2.lower().replace('_', ' ').split())
        vectorizer=TfidfVectorizer(analyzer='char', ngram_range=(2,3))
        tfidf_matrix=vectorizer.fit_transform([col1_tokens, col2_tokens])
        return cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:2])[0][0]
    
    def cosine_similarity_values(self, series1, series2):
        if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
            #numerical
            v1=(series1-series1.mean())/series1.std() if series1.std()>0 else series1-series1.mean()
            v2=(series2-series2.mean())/series2.std() if series2.std()>0 else series2-series2.mean()
            min_len=min(len(v1), len(v2))
            v1,v2=v1[:min_len], v2[:min_len]
            dot_product=np.dot(v1,v2)
            norm1,norm2=np.linalg.norm(v1), np.linalg.norm(v2)
            return dot_product/(norm1*norm2) if norm1>0 and norm2>0 else 0
        else:
            #categorical
            all_values=set(series1.unique()) | set(series2.unique())
            v1=[series1.value_counts().get(val,0) for val in all_values]
            v2=[series2.value_counts().get(val,0) for val in all_values]
            dot_product=np.dot(v1,v2)
            norm1,norm2=np.linalg.norm(v1), np.linalg.norm(v2)
            return dot_product/(norm1*norm2) if norm1>0 and norm2>0 else 0
        
    def sampled_jaccard_similarity(self, series1, series2):
        unique1=set(series1.dropna().unique())
        unique2=set(series2.dropna().unique())
        if len(unique1)>self.sample_size:
            unique1=set(np.random.choice(list(unique1),self.sample_size,replace=False))
        if len(unique2)>self.sample_size:
            unique2=set(np.random.choice(list(unique2),self.sample_size,replace=False))
        intersection=len(unique1&unique2)
        union=len(unique1|unique2)
        return intersection/union if union>0 else 0
    
    def cardinality_similarity(self, series1, series2):
        card1,card2=series1.nunique(),series2.nunique()
        max_card=max(card1,card2)
        min_card=min(card1,card2)
        return min_card/max_card if max_card>0 else 1

    def dtype_similarity(self, series1, series2):
        dtype_heirarchy={
            'int64':'numeric',
            'float64':'numeric',
            'int32':'numeric',
            'float32':'numeric',
            'object':'categorical',
            'string':'categorical',
            'category':'categorical',
            'datetime64[ns]':'temporal',
            'bool':'boolean'
        }
        type1=dtype_heirarchy.get(str(series1.dtype), 'other')
        type2=dtype_heirarchy.get(str(series2.dtype), 'other')
        if type1==type2:
            return 1.0
        elif (type1=='numeric' and type2=='boolean') or (type1=='boolean' and type2=='numeric'):
            return 0.7
        else:
            return 0.0
    
    def _compute_composite_score(self,edge_features):
        weights=self.thresholds['weight']
        return sum(edge_features[metric]*weights[metric] for metric in edge_features)
    
class FeatureTokenizer:
    def __init__(self, model_manager):
        self.model_manager=model_manager
        self.model=model_manager.model
        self.tokenizer=model_manager.tokenizer
        self.device=model_manager.device
        self.separator=" || "
    def tokenize_column_stats(self,stats_dict):
        column_text=self._stats_to_text(stats_dict)
        tokens=self._tokenize_text(column_text)
        embeddings=self._create_embeddings(tokens)
        return embeddings
    def _stats_to_text(self,stats_dict):
        inner_stats=list(stats_dict.values())[0]
        text_parts=[f"{k}:{v}" for k,v in inner_stats.items()]
        return self.separator.join(text_parts)
    def _tokenize_text(self,text):
        tokens=self.tokenizer(
            text,
            padding=True,
            return_tensors='pt'
        )
        tokens={k:v.to(self.device) for k,v in tokens.items()}
        return tokens
    def _create_embeddings(self, tokens):
        with torch.no_grad:
            embeddings=self.model.get_input_embeddings()(tokens['input_ids'])
        return embeddings
    def _pad_feature_list(self, feature_list):
        #have to pad them anyway
        if not feature_list:
            return torch.empty(0)
        max_length=max(features.shape[1] for features in feature_list)
        embedding_dim=self.model.get_input_emebddings().embedding_dim
        padded_features=[]
        for features in feature_list:
            current_length=features.shape[1]
            if current_length<max_length:
                padding=torch.zeros(1,
                                    max_length-current_length,
                                    embedding_dim,
                                    device=self.device)
                padded=torch.cat([features,padding], dim=1)
            else:
                padded=features
            padded_features.append(padded.squeeze(0))
        return torch.stack(padded_features)


    

def generate_vllm(prompt):
    return llm.generate(prompt, vllm_llm.sampling_params)


def table_to_nx_graph(df):
    G=nx.Graph()
    node_features=[]
    max_len=0
    for col in df.columns:
        col_stats=get_col_stats(df, col)
        G.add_node(f"col_{col}", node_type="column", col_stats)



