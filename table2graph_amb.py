import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import torch
import pandas as pd
import numpy as np
import itertools
from llm_serving.llm_call import generate_batch_without_decode
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
                    column_types[col]='continuous_numeric'
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
            columns=df.columns.tolist()
        batch_stats={}
        for col in columns:
            batch_stats[col]=self.get_col_stats(df, col)
        return batch_stats  

class ColumnContentExtractor:
    def __init__(self, sample_size=50):
        self.sample_size=sample_size
        self.content_template="Column:{header} || DataType: {dtype} || Content: {content}"
    def get_col_stats(self, df, col):
        series=df[col]
        sample_content=self._comprehensive_sample(series)
        formatted_content=self._format_content_for_tokenization(col, series, sample_content)
        return {
            f"column_name {col}": {
                "column_content":formatted_content,
                "column_header":col,
                "sample_size":str(len(sample_content)),
                "data_type":str(series.dtype)
            }
        }
    def _comprehensive_sample(self, series):
        sample=[]
        clean_series=series.dropna()
        #Include Null Representations [1-2 slots]
        if series.isnull().any():
            null_count=min(2, self.sample_size//10)
            sample.extend(["<NULL>"]*null_count)
        if len(clean_series)==0:
            return ["<NULL>"]*self.sample_size
        #Include Statistical Extremes [3-5 slots]
        if pd.api.types.is_numeric_dtype(series):
            extremes=[
                clean_series.min(),
                clean_series.max(),
                clean_series.median()
            ]
            if self.sample_size>=40:
                extremes.extend([
                    clean_series.quantile(0.25),
                    clean_series.quantile(0.75)
                ])
            sample.extend(extremes)
        #Most Frequent Values [30% of remaining slots]
        remaining_slots=self.sample_size-len(sample)
        frequent_slots=max(1, int(remaining_slots*0.3))
        value_counts=clean_series.value_counts()
        frequent_values=value_counts.head(frequent_slots).index.tolist()
        sample.extend(frequent_values)
        #Random Sample From Non Frequent Value [50% of remaining slots]
        remaining_slots=self.sample_size-len(sample)
        random_slots=max(1, int(remaining_slots*0.5))
        non_frequent=clean_series[~clean_series.isin(frequent_values)]
        if len(non_frequent)>0:
            random_sample_size=min(random_slots, len(non_frequent))
            random_values=non_frequent.sample(random_sample_size, random_state=42).tolist()
            sample.extend(random_values)
        #Fill Remaining Slots with most common value
        while len(sample)<self.sample_size:
            if len(value_counts)>0:
                sample.append(value_counts.index[0])
            else:
                sample.append("<EMPTY>")
        return sample[:self.sample_size]
    #ToDo: Implement other sampling methods, but lets stick to comprehensive for now
    def _format_content_for_tokenization(self, col_name, series, sample_content):
        formatted_values=[]
        for value in sample_content:
            if pd.isna(value) or value=="<NULL>":
                formatted_values.append("<NULL>")
            elif isinstance(value, (int, float)):
                formatted_values.append(str(value))
            else:
                clean_value=' '.join(str(value).split())[:100]
                formatted_values.append(clean_value)
        content_string=" | ".join(formatted_values)
        return self.content_template.format(
            header=col_name,
            dtype=str(series.dtype),
            content=content_string
        )
    def get_batch_stats(self,df,columns=None):
        if columns is None:
            columns=df.columns.tolist()
        batch_content={}
        for col in columns:
            batch_content[col]=self.get_col_stats(df, col)
        return batch_content

        


class RelationshipGenerator:
    def __init__(self, model_manager, threshold_config=None):
        self.model=model_manager.model
        self.tokenizer=model_manager.tokenizer
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
        columns=df.columns.tolist()
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                edge_features=self.compute_edge_features(df, col1, col2)
                composite_score=self._compute_composite_score(edge_features)
                if composite_score>=self.thresholds['composite_threshold']:
                    relationships.append({
                        'col1':col1, #why col1 and col2?
                        'col2':col2,
                        'edge_features':edge_features,
                        'composite_score':composite_score
                    })
        return relationships
    def compute_edge_features(self, df, col1, col2):
        return {
            'name_similarity':self.cosine_similarity_names(col1, col2),
            'value_similarity':self.cosine_similarity_values(df[col1], df[col2]),
            'jaccard_overlap':self.sampled_jaccard_similarity(df[col1], df[col2]),
            'cardinality_similarity':self.cardinality_similarity(df[col1], df[col2]),
            'dtype_similarity':self.dtype_similarity(df[col1], df[col2])
        }
    def compute_labeled_relationships(self, df, label_generator):
        relationships=[]
        columns=df.columns.tolist()
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                raw_features=self.compute_edge_features(df, col1, col2)
                feature_label=label_generator.generate_feature_label(raw_features)
                relationships.append({
                    'col1':col1,
                    'col2':col2,
                    'feature_label':feature_label, #GNN PREDICTION TARGET
                    'semantic_meaning':label_generator.get_semantic_interpretation(feature_label),
                    'auxiliary_features':{'name_similarity':raw_features['name_similarity'],
                    'cardinality_similarity':raw_features['cardinality_similarity']
                    }
                })
        return relationships
    def cosine_similarity_names(self, col1, col2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer=TfidfVectorizer(analyzer='char', ngram_range=(2,3))
        #Using a model's tokenizer, not sure if the semantic benefits are justified against the computational overhead
        #col1_tokens=self.tokenizer(col1, return_tensors='pt', padding=True)
        #col2_tokens=self.tokenizer(col2, return_tensors='pt', padding=True)
        # with torch.no_grad():
        #     col1_embeds=self.model.get_input_embeddings()(col1_tokens['input_ids']).mean(dim=1) #we are taking a mean of this value, why?
        #     col2_embeds=self.model.get_input_embeddings()(col2_tokens['input_ids']).mean(dim=1)
        vectors=vectorizer.fit_transform([col1, col2])
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    
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
        weights=self.thresholds['weights']
        return sum(edge_features[metric]*weights[metric] for metric in edge_features)
    
class SemanticLabelGenerator:
    def __init__(self):
        self.intervals={
            'low':(0.00, 0.33),
            'medium':(0.34, 0.67),
            'high':(0.68, 1.00)
        }
        self.semantic_ontology=self._create_18_label_ontology()
    def extract_core_features(self, edge_features):
        return {
            'cosine_relationality':edge_features['value_similarity'],
            'jaccard_sampling':edge_features['jaccard_overlap'],
            'same_dtype': 1 if edge_features['dtype_similarity']==1.0 else 0
        }
    def discretize_features(self, value, feature_name):
        if feature_name=='same_dtype':
            return 'NUM' if value==1 else 'CAT'
        for interval, (low, high) in self.intervals.items():
            if low<=value<=high:
                return interval.upper()
        return 'LOW' #fallback uhhh, judas gonna be mine
    def generate_feature_label(self, edge_features):
        core=self.extract_core_features(edge_features)
        cos_interval=self.discretize_features(core['cosine_relationality'], 'cosine')
        jac_interval=self.discretize_features(core['jaccard_sampling'], 'jaccard')
        dtype_flag=self.discretize_features(core['same_dtype'], 'same_dtype')
        return f"{cos_interval}_COS_{jac_interval}_JAC_{dtype_flag}"
    def get_semantic_interpretation(self, feature_label):
        return self.semantic_ontology.get(feature_label, "UNKNOWN_RELATIONSHIP")
    def _create_18_label_ontology(self):    #EXTREMELY IMPORTANT, EVERYTHING HINGES ON THESE DEFINITIONS, HEART OF SEMANTIC EMERGENCE
        return {
            # HIGH COSINE (STRONG DIRECTIONAL RELATIONSHIP)
            "HIGH_COS_HIGH_JAC_NUM":"LINEAR_NUMERICAL_DEPENDENCE_WITH_SHARED_VALUES",
            "HIGH_COS_HIGH_JAC_CAT":"IDENTICAL_OR_REDUNDANT",
            "HIGH_COS_MEDIUM_JAC_NUM":"COMPUTATIONAL_DEPENDENCE_WITH_SHARED_VALUES",
            "HIGH_COS_MEDIUM_JAC_CAT":"STRONG_CATEGORICAL_RELATIONSHIP_PARTIAL_OVERLAP",
            "HIGH_COS_LOW_JAC_NUM":"LINEAR_NUMERICAL_DEPENDENCE",
            "HIGH_COS_LOW_JAC_CAT":"STRONG_CATEGORICAL_RELATIONSHIP_DISTINCT_VALUES",
            # MEDIUM COSINE (MODERATE RELATIONSHIP)
            "MEDIUM_COS_HIGH_JAC_NUM":"NUMERICAL_ALIAS_WITH_NOISE",
            "MEDIUM_COS_HIGH_JAC_CAT":"MODERATE_CATEGORICAL_OVERLAP",
            "MEDIUM_COS_MEDIUM_JAC_NUM":"MODERATE_NUMERICAL_CORRELATION_SHARED_VALUES",
            "MEDIUM_COS_MEDIUM_JAC_CAT":"PARTIAL_CATEGORICAL_OVERLAP",
            "MEDIUM_COS_LOW_JAC_NUM":"MODERATE_NUMERICAL_CORRELATION",
            "MEDIUM_COS_LOW_JAC_CAT":"WEAK_CATEGORICAL_RELATIONSHIP",
            # LOW COSINE (WEAK/NO DIRECTIONAL RELATIONSHIP)
            "LOW_COS_HIGH_JAC_NUM":"NUMERICAL_IDENTITY_NO_CORRELATION",
            "LOW_COS_HIGH_JAC_CAT":"CATEGORICAL_DERIVATION_OR_ALIAS",
            "LOW_COS_MEDIUM_JAC_NUM":"SHARED_NUMERICAL_VALUES_NO_CORRELATION",
            "LOW_COS_MEDIUM_JAC_CAT":"PARTIAL_CATEGORICAL_OVERLAP_NO_CORRELATION",
            "LOW_COS_LOW_JAC_NUM":"WEAK_NUMERICAL_RELATIONSHIP",
            "LOW_COS_LOW_JAC_CAT":"WEAK_HETEROGENEOUS_RELATIONSHIP"
        }
    
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
    def tokenize_semantic_label(self, label_string):
        text=f"relationship_type: {label_string}"
        tokens=self._tokenize_text(text)
        return self._create_embeddings(tokens)
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
        with torch.no_grad():
            embeddings=self.model.get_input_embeddings()(tokens['input_ids'])
        return embeddings
    def _pad_feature_list(self, feature_list):
        #have to pad them anyway
        if not feature_list:
            return torch.empty(0)
        max_length=max(features.shape[1] for features in feature_list)
        embedding_dim=self.model.get_input_embeddings().embedding_dim
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
    def batch_tokenize_columns(self, df, stats_extractor):
        feature_list=[]
        for col in df.columns:
            col_stats=stats_extractor.get_col_stats(df,col)
            embeddings=self.tokenize_column_stats(col_stats)
            feature_list.append(embeddings)
        return self._pad_feature_list(feature_list)
    def batch_tokenize_semantic_labels(self, label_list):
        if not label_list:
            return torch.empty(0)
        label_embeddings=[]
        for label in label_list:
            embedding=self.tokenize_semantic_label(label)
            label_embeddings.append(embedding)
        return self._pad_feature_list(label_embeddings)

class LightweightFeatureTokenizer:
    def __init__(self, embedding_strategy='hybrid'):
        self.embedding_strategy=embedding_strategy
        self.semantic_encoder=None
        self.vectorizer=None
        self.feature_dim=512
        self._initialize_encoders()
    def _initialize_encoders(self):
        if 'semantic' in self.embedding_strategy:
            from sentence_transformers import SentenceTransformer
            self.semantic_encoder=SentenceTransformer('all-MiniLM-L6-v2')
        if 'statistical' in self.embedding_strategy:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer=TfidfVectorizer(max_features=256, ngram_range=(1,2))
    def encode_column_content(self, content_dict):
        inner_content=list(content_dict.values())[0]
        content_text=inner_content['column_content']
        dtype=inner_content['data_type']
        sample_size=int(inner_content['sample_size']) # column header is not explicitly being stored
        embeddings=[]
        if self.semantic_encoder:
            semantic_embed=self.semantic_encoder.encode(content_text)
            embeddings.append(semantic_embed)
        if self.vectorizer:
            if hasattr(self.vectorizer, 'vocabulary_'):
                statistical_embed=self.vectorizer.transform([content_text]).toarray()[0]
                embeddings.append(statistical_embed)
        metadata_features=self._engineer_metadata_features(dtype, sample_size, content_text)
        embeddings.append(metadata_features)
        if embeddings:
            return np.concatenate(embeddings)
        else:
            return np.zeros(self.feature_dim)
    def _engineer_metadata_features(self, dtype, sample_size, content_text):
        features=[
            1.0 if 'int' in dtype else 0.0,
            1.0 if 'float' in dtype else 0.0,
            1.0 if 'object' in dtype else 0.0,
            sample_size/100.0, #Normalised sample size
            content_text.count('<NULL>')/sample_size, #Null ratio
            len(content_text)/1000.0, #Content length
            content_text.count('|')/sample_size, #Value diversity
            1.0 if any(c.isdigit() for c in content_text) else 0.0, #Contains numbers
        ]
        return np.array(features)
    
class GraphBuilder:
    def __init__(self, content_extractor, feature_tokenizer, relationship_generator, semantic_label_generator, mode='train'):
        self.content_extractor=content_extractor
        self.feature_tokenizer=feature_tokenizer
        self.relationship_generator=relationship_generator
        self.semantic_label_generator=semantic_label_generator
        self.mode=mode
        #Classification Setup
        self.label_to_index=self._create_label_mapping()
        self.index_to_label={v:k for k,v in self.label_to_index.items()}
        self.num_classes=len(self.label_to_index)
    def _create_label_mapping(self):
        ontology=self.semantic_label_generator.semantic_ontology
        return {label: idx for idx, label in enumerate(ontology.keys())}
    def build_graph(self, df):
        #Main Orchestration Method, returns torch_geometric data object for GNN processing
        #1: Create embedded nodes for columns
        node_features, node_mapping=self._create_embedded_nodes(df)
        if self.mode=='train':
            edge_index, edge_labels=self._create_supervised_edges(df, node_mapping)
        else: #Test Mode
            edge_index=self._create_candidate_edges(df, node_mapping)
            edge_labels=None #Will be predicted
        return self._to_pytorch_geometric(node_features, edge_index, edge_labels)
    def _create_embedded_nodes(self, df):
        node_features=[]
        node_mapping={} #column_name -> node_index
        for idx, col in enumerate(df.columns):
            content_dict=self.content_extractor.get_col_stats(df, col)
            embedding=self.feature_tokenizer.encode_column_content(content_dict)
            node_features.append(embedding)
            node_mapping[col]=idx
        return torch.stack([torch.tensor(f, dtype=torch.float32) for f in node_features]), node_mapping
    def _create_supervised_edges(self, df, node_mapping):
        #Creating edges with ground truth labels for supervision
        relationships=self.relationship_generator.compute_labeled_relationships(df, self.semantic_label_generator)
        edge_index=[]
        edge_labels=[]
        for rel in relationships:
            #Sparse Connectivity to avoid emulating a MLP cosplaying as a graph
            if self._passes_thresholds(rel):
                src_idx=node_mapping[rel['col1']]
                dst_idx=node_mapping[rel['col2']]
                #Undirected Edges: Add both directions
                edge_index.extend([[src_idx, dst_idx], [dst_idx, src_idx]])
                #Convert Semantic Label to classification index
                label_idx=self.label_to_index[rel['feature_label']]
                edge_labels.extend([label_idx, label_idx])
        return torch.tensor(edge_index).T, torch.tensor(edge_labels, dtype=torch.long)
    def _create_candidate_edges(self, df, node_mapping):
        #Create candidate edges for predictions
        relationships=self.relationship_generator.compute_all_relationship_scores(df)
        edge_index=[]
        for rel in relationships:
            #Same threshold logic as training
            src_idx=node_mapping[rel['col1']]
            dst_idx=node_mapping[rel['col2']]
            #Undirected edges
            edge_index.extend([src_idx, dst_idx], [dst_idx, src_idx])
        return torch.tensor(edge_index).T if edge_index else torch.empty((2,0), dtype=torch.long)
    def _passes_threshold(self, relationship):
        return relationship.get('composite_score', 0)>=self.relationship_generator.thresholds['composite_threshold']
    def _to_pytorch_geometric(self, node_features, edge_index, edge_labels=None):
        data=Data(x=node_features, edge_index=edge_index)
        if edge_labels is not None:
            data.edge_attr=edge_labels
        return data
class GNNEdgePredictor(torch.nn.Module):
    #3 Layer GNN to capture direct + transitive relationships
    #Edge prediction via node embedding concatenation
    #Separate Classifier Head
    #Built-in training and test steps
    def __init__(self, node_dim, hidden_dim=256, num_classes=18, num_layers=3, dropout=0.1):
        super().__init__()
        #3-hop GNN
        self.gnn=TableGCN(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers
        )
        #Edge Classifier: Concatenated Node Embeds -> Edge Labels
        self.edge_classifier=torch.nn.Sequential(torch.nn.Linear(hidden_dim*2, hidden_dim),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Dropout(dropout),
                                                 torch.nn.Linear(hidden_dim, hidden_dim//2),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Dropout(dropout),
                                                 torch.nn.Linear(hidden_dim//2, num_classes)
                                                 )
        self.criteion=torch.nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.parameters(), lr=0.01) #Tweak Tweak
    def forward(self, pyg_data):
        #Forward Pass: Node Features -> GNN -> Edge Predictions
        # ---
        #Update Node Embeds (3-hop)
        node_embeddings=self.gnn(pyg_data.x, pyg_data.edge_index)
        #Get source and dest embeds
        src_embeddings=node_embeddings[pyg_data.edge_index[0]]
        dst_embeddings=node_embeddings[pyg_data.edge_index[1]]
        #Concat for edge classification
        edge_embeddings=torch.cat([src_embeddings, dst_embeddings], dim=1)
        #Classify edge types
        edge_logits=self.edge_classifier(edge_embeddings)
        return edge_logits
    def train_step(self, pyg_data):
        self.train()
        self.optimizer.zero_grad()
        edge_logits=self.forward(pyg_data)
        loss=self.criteion(edge_logits, pyg_data.edge_attr)
        loss.backward()
        self.optimizer.step()
        #Calculate accuracy for monitoring
        predictions=torch.argmax(edge_logits, dim=1)
        accuracy=(predictions==pyg_data.edge_attr).float().mean() #Pooling!!
        return loss.item(), accuracy.item()
    def predict(self, pyg_data):
        self.eval()
        with torch.no_grad():
            edge_logits=self.forward(pyg_data)
            predictions=torch.argmax(edge_logits, dim=1)
            confidences=torch.softmax(edge_logits, dim=1)
            max_confidences=torch.max(confidences, dim=1)[0]
        return predictions, max_confidences
class Table2GraphPipeLine:
    def __init__(self, embedding_strategy='hybrid'):
        self.content_extractor=ColumnContentExtractor
        self.feature_tokenizer=LightweightFeatureTokenizer(embedding_strategy)
        self.relationship_generator=None #We keep this None for the classification task, maybe later when we move to more complex architectures which capture relationality we can think about embedding similarity
        self.semantic_label_generator=SemanticLabelGenerator()
        #Graph Builders
        self.train_builder=None
        self.test_builder=None
        self.predictor=None
    def initialize_for_training(self, model_manager=None, node_dim=512):
        #Init RelationshipGenerator if needed
        if self.relationship_generator is None:
            self.relationship_generator=RelationshipGenerator(model_manager, threshold_config={'composite_threshold':0.3})
        self.train_builder=GraphBuilder(
            self.content_extractor,
            self.feature_tokenizer,
            self.relationship_generator,
            self.semantic_label_generator,
            mode='train'
        )
        self.predictor=GNNEdgePredictor(
            node_dim=node_dim,
            hidden_dim=256,
            num_classes=self.train_builder.num_classes,
            num_layers=3
        )
    def initialise_for_testing(self):
        self.test_builder=GraphBuilder(
            self.content_extractor,
            self.feature_tokenizer,
            self.relationship_generator,
            self.semantic_label_generator,
            mode='test'
        )
    def train_epoch(self, table_dataframes):
        #Train on multiple tables for each epoch
        total_loss=0
        total_accuracy=0
        num_batches=0
        for df in table_dataframes:
            try:
                pyg_data=self.train_builder.build_graph(df)
                if pyg_data.edge_index.size(1)>0:
                    loss, accuracy = self.predictor.train_step(pyg_data)
                    total_loss+=loss
                    total_accuracy+=accuracy
                    num_batches+=1
            except Exception as e:
                print(f"Warning: Skipped Table due to error {e}")
        avg_loss=total_loss/max(num_batches, 1)
        avg_accuracy=total_accuracy/max(num_batches, 1)
        return avg_loss, avg_accuracy
    def predict_relationships(self, df):
        pyg_data=self.test_builder.build_graph(df)
        if pyg_data.edge_index.size(1)==0:
            return [] #No candidate edges
        predictions, confidences=self.predictor.predict(pyg_data)
        #Convert back to semantic labels ??
        results=[]
        columns=list(df.columns)
        #Process Edges
        processed_pairs=set()
        for i, (src_idx, dst_idx) in enumerate(pyg_data.edge_index.T):
            src_col=columns[src_idx.item()]
            dst_col=columns[dst_idx.item()]
            #Skip if already processed
            pair_key=tuple(sorted([src_col, dst_col]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            #Convert Prediction to Semantic Label
            label_idx=predictions[i].item()
            confidence=confidences[i].item()
            semantic_label=self.test_builder.index_to_label[label_idx]
            results.append({
                'col1': src_col,
                'col2':dst_col,
                'predicted_label':semantic_label,
                'confidence':confidence,
                'semantic_meaning':self.semantic_label_generator.get_semantic_interpretation(semantic_label)
            })
        return results




#ToDo -Run through additions, fix TableGCN, figure out running pipeline, data ingestion (also involves data collection)



