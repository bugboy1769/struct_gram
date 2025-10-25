"""
Contrastive Learning Pipeline for Table-to-Graph Semantic Analysis
====================================================================

This file implements a contrastive learning approach where:
- Table structures are converted to graph embeddings using GNNs
- Graph embeddings are aligned with question embeddings using InfoNCE loss
- Edges are created based on computed features (sparse graph construction)
- Training optimizes table-question alignment rather than edge classification

This is a refactor of table2graph_sem.py to support the contrastive learning paradigm.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader
from gcn_conv import TableGCN
from pathlib import Path
from sentence_transformers import SentenceTransformer

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

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

    def preprocess_columns(self, df):
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

# ============================================================================
# COLUMN CONTENT EXTRACTION
# ============================================================================

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

# ============================================================================
# RELATIONSHIP FEATURE COMPUTATION
# ============================================================================

class RelationshipGenerator:
    def __init__(self, threshold_config=None):
        self.thresholds=threshold_config or {
            'composite_threshold':0.4,
            'weights': {
                'name_similarity':0.10,
                'value_similarity':0.15,
                'jaccard_overlap':0.1,
                'cardinality_similarity':0.05,
                'dtype_similarity':0.05,
                #Semantic Features
                'id_reference':0.15,
                'hierarchical':0.10,
                'functional_dependency':0.10,
                'measure_dimension':0.10,
                'temporal_dependency':0.10
            }
        }
        self.sample_size=100
    def compute_all_relationship_scores(self, df, stats_dict=None):
        relationships=[]
        columns=df.columns.tolist()
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                edge_features=self.compute_edge_features(df, col1, col2)
                composite_score=self._compute_composite_score(edge_features)
                if composite_score>=self.thresholds['composite_threshold']:
                    relationships.append({
                        'col1':col1,
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
            'dtype_similarity':self.dtype_similarity(df[col1], df[col2]),
            'id_reference':self.detect_id_reference_pattern(df[col1], df[col2]),
            'hierarchical':self.detect_hierarchical_pattern(df[col1], df[col2]),
            'functional_dependency':self.detect_functional_dependency(df[col1], df[col2]),
            'measure_dimension':self.detect_measure_dimension_pattern(df[col1], df[col2]),
            'temporal_dependency':self.detect_temporal_dependency(df[col1], df[col2])
        }
    def compute_labeled_relationships(self, df, label_generator):
        relationships=[]
        columns=df.columns.tolist()
        all_composite_scores = []
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                raw_features=self.compute_edge_features(df, col1, col2)
                composite_score = self._compute_composite_score(raw_features)
                all_composite_scores.append(composite_score)
                feature_label=label_generator.generate_feature_label(raw_features)
                relationships.append(
                    {
                    'col1':col1,
                    'col2':col2,
                    'feature_label':feature_label,
                    'composite_score':composite_score,
                    'semantic_meaning':label_generator.get_semantic_interpretation(feature_label),
                    'auxiliary_features':{
                        'name_similarity':raw_features['name_similarity'],
                        'cardinality_similarity':raw_features['cardinality_similarity'],
                        'id_reference': raw_features['id_reference'],
                        'hierarchical':raw_features['hierarchical'],
                        'functional_dependency':raw_features['functional_dependency'],
                        'measure_dimension':raw_features['measure_dimension'],
                        'temporal_dependency':raw_features['temporal_dependency'],
                        }
                })
        if all_composite_scores:
            print(f"\n[Composite Score Stats for {df.shape}]")
            print(f"  Min:    {min(all_composite_scores):.4f}")
            print(f"  Max:    {max(all_composite_scores):.4f}")
            print(f"  Mean:   {np.mean(all_composite_scores):.4f}")
            print(f"  Median: {np.median(all_composite_scores):.4f}")
            print(f"  Threshold: {self.thresholds['composite_threshold']:.4f}")
            above_threshold = sum(1 for s in all_composite_scores if s >= self.thresholds['composite_threshold'])
            print(f"  Above threshold: {above_threshold}/{len(all_composite_scores)} ({above_threshold/len(all_composite_scores)*100:.1f}%)")
        return relationships
    def cosine_similarity_names(self, col1, col2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer=TfidfVectorizer(analyzer='char', ngram_range=(2,3))
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
            vc1 = series1.value_counts().to_dict()
            vc2 = series2.value_counts().to_dict()
            v1=[vc1.get(val, 0) for val in all_values]
            v2=[vc2.get(val, 0) for val in all_values]
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
    def detect_id_reference_pattern(self, series1, series2):
        if series1.dtype=='float64' and series2.dtype=='float64':
            return 0.0
        s1_clean=series1.dropna()
        s2_clean=series2.dropna()
        if len(s1_clean)==0 or len(s2_clean)==0:
            return 0.0
        s1_unique=set(s1_clean.unique())
        s2_unique=set(s2_clean.unique())
        if len(s2_unique)==0:
            return 0.0
        #Check for containment: s2 in s1
        containment_ratio=len(s2_unique & s1_unique)/len(s2_unique)
        #Check for cardinality ratio: s1.unique > s2.unique for FK reln
        cardinality_ratio=len(s1_unique)/max(len(s2_unique), 1)
        #Boost score if ID patterns
        col1_name=str(series1.name).lower() if series1.name else ""
        col2_name=str(series2.name).lower() if series2.name else ""
        name_boost=1.0
        if ('id' in col1_name and 'id' in col2_name) or (col1_name.endswith('_id') or col2_name.endswith('_id')):
            name_boost=1.3
        #FK Score: High containment + good cardinality ratio
        if containment_ratio>0.7 and cardinality_ratio>1.5:
            return min(1.0, containment_ratio*min(cardinality_ratio/10, 1.0)*name_boost)
        return containment_ratio*0.3
    def detect_hierarchical_pattern(self, series1, series2):
        if not (pd.api.types.is_string_dtype(series1) or pd.api.types.is_object_dtype(series1)) or not (pd.api.types.is_string_dtype(series2) or pd.api.types.is_object_dtype(series2)):
            return 0.0
        s1_sample=series1.dropna().astype(str).sample(min(50, len(series1.dropna())), random_state=42)
        s2_sample=series2.dropna().astype(str).sample(min(50, len(series2.dropna())), random_state=42)
        if len(s1_sample)==0 or len(s2_sample)==0:
            return 0.0
        #Check if values in one column are substrings or prefixes of other
        substring_s1_in_s2=0
        substring_s2_in_s1=0
        for v1 in s1_sample:
            if any(str(v1).lower() in str(v2).lower() for v2 in s2_sample):
                substring_s1_in_s2+=1
        for v2 in s2_sample:
            if any(str(v2).lower() in str(v1).lower() for v1 in s1_sample):
                substring_s2_in_s1+=1
        s1_substring_score=substring_s1_in_s2/len(s1_sample)
        s2_substring_score=substring_s2_in_s1/len(s2_sample)
        #Check cardinality heirarchy
        card1, card2 = series1.nunique(), series2.nunique()
        heirarchy_score=0.0
        if card1<card2 and card1>0:
            heirarchy_score=min(card2/card1, 10)/10
        elif card2<card1 and card2>0:
            heirarchy_score=min(card1/card2, 10)/10
        return max(s1_substring_score, s2_substring_score)*0.7 + heirarchy_score*0.3
    def detect_functional_dependency(self, series1, series2):
        df_temp=pd.DataFrame({'s1':series1, 's2':series2}).dropna()
        if len(df_temp)<5:
            return 0.0
        try:
            grouped=df_temp.groupby('s1')['s2'].nunique()
            if len(grouped)==0:
                return 0.0
            functional_dependency_score=(grouped==1).sum()/len(grouped)
            s1_unique, s2_unique=series1.nunique(), series2.nunique()
            if s1_unique>0 and s2_unique>0:
                cardinality_factor=min(s1_unique/s2_unique, 5)/5
                return functional_dependency_score*(0.8+0.2*cardinality_factor)
            return functional_dependency_score
        except Exception:
            return 0.0
    def detect_measure_dimension_pattern(self, series1, series2):
        #Check for numeric-categorical pair
        s1_numeric=pd.api.types.is_numeric_dtype(series1)
        s2_numeric=pd.api.types.is_numeric_dtype(series2)
        if s1_numeric==s2_numeric:
            return 0.0
        #Identify measure v dimension
        if s1_numeric:
            measure_col, dimension_col=series1, series2
        else:
            measure_col, dimension_col=series2, series1
        #Remove nulls
        df_temp=pd.DataFrame({'measure':measure_col, 'dimension':dimension_col}).dropna()
        if len(df_temp)<5:
            return 0.0
        try:
            #Check aggregation potential
            grouped_stats=df_temp.groupby('dimension')['measure'].agg(['count', 'std', 'mean'])
            if len(grouped_stats)<2:
                return 0.0
            avg_count_per_group=grouped_stats['count'].mean()
            std_of_means=grouped_stats['mean'].std()
            overall_std=df_temp['measure'].std()
            #Scoring
            aggregation_potential=min(avg_count_per_group/3, 1.0)
            variation_score=(std_of_means/overall_std) if overall_std>0 else 0
            cardinality_score=1.0 if 2<= dimension_col.nunique()<=20 else 0.5
            return (aggregation_potential*0.4+variation_score*0.4+cardinality_score*0.2)
        except Exception:
            return 0.0
    def detect_temporal_dependency(self, series1, series2):
        s1_datetime=pd.api.types.is_datetime64_any_dtype(series1)
        s2_datetime=pd.api.types.is_datetime64_any_dtype(series2)


        if not s1_datetime and pd.api.types.is_object_dtype(series1):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(series1.dropna().head(10))
                s1_datetime=True
            except:
                pass
        if not s2_datetime and pd.api.types.is_object_dtype(series2):
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(series2.dropna().head(10))
                s2_datetime=True
            except:
                pass
        if not (s1_datetime and s2_datetime):
            if pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2):
                return self._detect_numeric_sequence_correlation(series1, series2)
            return 0.0
        try:
            #Convert to datetime if needed
            if s1_datetime and not pd.api.types.is_datetime64_any_dtype(series1):
                dt1=pd.to_datetime(series1, errors='coerce')
            else:
                dt1=series1
            if s2_datetime and not pd.api.types.is_datetime64_any_dtype(series2):
                dt2=pd.to_datetime(series2, errors='coerce')
            else:
                dt2=series2
            df_temp=pd.DataFrame({'dt1':dt1, 'dt2':dt2}).dropna()
            if len(df_temp)<5:
                return 0.0
            #Check for temporal corr
            correlation=df_temp['dt1'].astype('int64').corr(df_temp['dt2'].astype('int64'))
            #Check for consistent time gaps
            time_diffs=(df_temp['dt2']-df_temp['dt1']).dt.total_seconds()
            consistent_gap=time_diffs.std()/(time_diffs.mean() + 1e-10) if len(time_diffs)>1 else 1
            #Combine corr and consistency
            temporal_score=abs(correlation)*0.7+(1/(1+consistent_gap))*0.3
            return min(temporal_score, 1.0)
        except Exception:
            return 0.0
    def _detect_numeric_sequence_correlation(self, series1, series2):
        try:
            df_temp=pd.DataFrame({'s1':series1, 's2':series2}).dropna()
            if len(df_temp)<5:
                return 0.0
            correlation=df_temp['s1'].corr(df_temp['s2'])
            s1_diffs=df_temp['s1'].diff().dropna()
            s2_diffs=df_temp['s2'].diff().dropna()
            if len(s1_diffs)>0 and len(s2_diffs)>0:
                diff_correlation=s1_diffs.corr(s2_diffs)
                return abs(correlation)*0.6 + abs(diff_correlation)*0.4
            return abs(correlation)*0.8
        except Exception:
            return 0.0

    def _compute_composite_score(self,edge_features):
        weights=self.thresholds['weights']
        return sum(edge_features[metric]*weights.get(metric, 0.0) for metric in edge_features if metric in weights)

# ============================================================================
# SEMANTIC LABEL GENERATOR (For Question Generation)
# ============================================================================

class SemanticLabelGenerator:
    def __init__(self):
        self.semantic_ontology=self._create_semantic_ontology()
    def generate_feature_label(self, edge_features):
        """
        Rule-based semantic classification using decision tree.
        Priority order: JOIN → TEMPORAL → AGGREGATION → DERIVATION → STRUCTURAL → FALLBACK
        """
        # Extract semantic features
        id_ref = edge_features.get('id_reference', 0.0)
        hier = edge_features.get('hierarchical', 0.0)
        func_dep = edge_features.get('functional_dependency', 0.0)
        meas_dim = edge_features.get('measure_dimension', 0.0)
        temp_dep = edge_features.get('temporal_dependency', 0.0)

        # Extract statistical features (supporting evidence)
        val_sim = edge_features.get('value_similarity', 0.0)
        jac_sim = edge_features.get('jaccard_overlap', 0.0)
        dtype_sim = edge_features.get('dtype_similarity', 0.0)
        card_sim = edge_features.get('cardinality_similarity', 0.0)
        name_sim = edge_features.get('name_similarity', 0.0)

        # ==================== PRIORITY 1: JOIN RELATIONSHIPS ====================
        if id_ref > 0.7 and func_dep > 0.7:
            return "PRIMARY_FOREIGN_KEY"

        if id_ref > 0.5 and func_dep > 0.5:
            return "FOREIGN_KEY_CANDIDATE"

        if id_ref > 0.6 and card_sim < 0.3:
            return "REVERSE_FOREIGN_KEY"

        if jac_sim > 0.7 and dtype_sim == 1.0 and val_sim > 0.5:
            return "NATURAL_JOIN_CANDIDATE"

        if jac_sim > 0.4 and dtype_sim > 0.7:
            return "WEAK_JOIN_CANDIDATE"

        if id_ref > 0.4 and dtype_sim < 0.5:
            return "CROSS_TABLE_REFERENCE"

        if jac_sim > 0.6 and func_dep < 0.3:
            return "MANY_TO_MANY_REFERENCE"

        if id_ref > 0.5 and func_dep > 0.3 and func_dep < 0.7:
            return "SELF_REFERENTIAL_KEY"

        # ==================== PRIORITY 2: TEMPORAL RELATIONSHIPS ====================
        if temp_dep > 0.7:
            return "TEMPORAL_SEQUENCE_STRONG"

        if temp_dep > 0.4:
            return "TEMPORAL_SEQUENCE_WEAK"

        if temp_dep > 0.3 and val_sim > 0.5:
            return "TEMPORAL_CORRELATION"

        # ==================== PRIORITY 3: AGGREGATION RELATIONSHIPS ====================
        if meas_dim > 0.7:
            return "MEASURE_DIMENSION_STRONG"

        if hier > 0.5 and meas_dim > 0.4:
            return "DIMENSION_HIERARCHY"

        if meas_dim > 0.4:
            return "MEASURE_DIMENSION_WEAK"

        if meas_dim > 0.3 and card_sim < 0.2:
            return "FACT_DIMENSION"

        if meas_dim > 0.3 and hier < 0.3:
            return "NATURAL_GROUPING"

        if hier > 0.4 and meas_dim > 0.3:
            return "NESTED_AGGREGATION"

        if meas_dim > 0.2 and card_sim < 0.15:
            return "PIVOT_CANDIDATE"

        # ==================== PRIORITY 4: DERIVATION RELATIONSHIPS ====================
        if val_sim > 0.9 and jac_sim > 0.8:
            return "REDUNDANT_COLUMN"

        if func_dep > 0.8 and val_sim < 0.3:
            return "FUNCTIONAL_TRANSFORMATION"

        if val_sim > 0.7 and name_sim < 0.3:
            return "DERIVED_CALCULATION"

        if hier > 0.5 and func_dep > 0.5:
            return "AGGREGATED_DERIVATION"

        if val_sim > 0.8 and dtype_sim == 1.0:
            return "NORMALIZED_VARIANT"

        if name_sim > 0.7 and jac_sim > 0.3:
            return "SYNONYM_COLUMN"

        # ==================== PRIORITY 5: STRUCTURAL RELATIONSHIPS ====================
        if func_dep > 0.6 and id_ref > 0.3:
            return "COMPOSITE_KEY_COMPONENT"

        if card_sim < 0.15 and func_dep > 0.4:
            return "PARTITION_KEY"

        if func_dep > 0.5 and jac_sim < 0.4:
            return "INDEX_CANDIDATE"

        if temp_dep > 0.2 and val_sim < 0.3:
            return "AUDIT_RELATIONSHIP"

        if temp_dep > 0.2 and func_dep > 0.3:
            return "VERSION_TRACKING"

        # ==================== PRIORITY 6: FALLBACK ====================
        if val_sim > 0.5 or jac_sim > 0.4:
            return "WEAK_CORRELATION"

        moderate_features = sum([
            id_ref > 0.3,
            hier > 0.3,
            func_dep > 0.3,
            meas_dim > 0.3,
            temp_dep > 0.3
        ])
        if moderate_features >= 3:
            return "AMBIGUOUS_RELATIONSHIP"

        return "INDEPENDENT_COLUMNS"

    def get_semantic_interpretation(self, feature_label):
        return self.semantic_ontology.get(feature_label, "UNKNOWN_RELATIONSHIP")
    def _create_semantic_ontology(self):
        return {
                    # ==================== CATEGORY 1: JOIN RELATIONSHIPS (8) ====================
        "PRIMARY_FOREIGN_KEY": "Strong foreign key relationship - PRIMARY JOIN candidate with high containment and functional dependency",
        "FOREIGN_KEY_CANDIDATE": "Likely foreign key relationship - NATURAL JOIN candidate with moderate ID reference pattern",
        "REVERSE_FOREIGN_KEY": "Reverse foreign key pattern - JOIN possible but reversed cardinality",

        "NATURAL_JOIN_CANDIDATE": "High value overlap with same data type - NATURAL JOIN or USING clause candidate",
        "WEAK_JOIN_CANDIDATE": "Some value overlap with compatible types - Possible JOIN with ON condition",
        "CROSS_TABLE_REFERENCE": "Moderate reference pattern across different types - Complex JOIN candidate",
        "MANY_TO_MANY_REFERENCE": "Low functional dependency with high overlap - Junction table pattern",
        "SELF_REFERENTIAL_KEY": "High ID reference within related context - Hierarchical self-JOIN pattern",

        # ==================== CATEGORY 2: AGGREGATION RELATIONSHIPS (7) ====================
        "MEASURE_DIMENSION_STRONG": "Strong measure-dimension relationship - PRIMARY GROUP BY with aggregation target",
        "MEASURE_DIMENSION_WEAK": "Moderate measure-dimension relationship - Secondary GROUP BY candidate",
        "DIMENSION_HIERARCHY": "Hierarchical categorical relationship - Nested GROUP BY or ROLLUP candidate",
        "FACT_DIMENSION": "Fact-dimension pattern - Star schema relationship for aggregation queries",

        "NATURAL_GROUPING": "Natural grouping pattern - Direct GROUP BY relationship",
        "NESTED_AGGREGATION": "Hierarchical with measure pattern - Multi-level GROUP BY with aggregations",
        "PIVOT_CANDIDATE": "Low cardinality categorical with numeric - PIVOT or CASE aggregation candidate",

        # ==================== CATEGORY 3: ORDERING RELATIONSHIPS (5) ====================
        "TEMPORAL_SEQUENCE_STRONG": "Strong temporal correlation - PRIMARY ORDER BY candidate for time-series",
        "TEMPORAL_SEQUENCE_WEAK": "Moderate temporal correlation - Secondary ORDER BY or time-based filtering",
        "TEMPORAL_CORRELATION": "Datetime columns with correlation - Time-based JOIN or ORDER BY candidate",

        "SEQUENTIAL_ORDERING": "Numeric sequence correlation - ORDER BY candidate for ranked queries",
        "RANKED_RELATIONSHIP": "Ordinal pattern detected - RANK or ROW_NUMBER partitioning candidate",

        # ==================== CATEGORY 4: DERIVATION RELATIONSHIPS (6) ====================
        "DERIVED_CALCULATION": "High similarity with different names - One column likely calculated from other",
        "FUNCTIONAL_TRANSFORMATION": "Strong functional dependency without overlap - Mathematical or string transformation",
        "AGGREGATED_DERIVATION": "Parent-child with aggregation pattern - Derived aggregate or summary column",

        "REDUNDANT_COLUMN": "Nearly identical content - Candidate for deduplication or normalization",
        "NORMALIZED_VARIANT": "Same semantic content, different encoding - Normalization or standardization variant",
        "SYNONYM_COLUMN": "High name similarity with compatible content - Potential synonym or alias column",

        # ==================== CATEGORY 5: STRUCTURAL RELATIONSHIPS (5) ====================
        "COMPOSITE_KEY_COMPONENT": "Part of composite key pattern - Multi-column uniqueness constraint",
        "PARTITION_KEY": "Low cardinality with high coverage - Table partitioning candidate",
        "INDEX_CANDIDATE": "High uniqueness with filtering potential - Index creation candidate",

        "AUDIT_RELATIONSHIP": "Temporal with low correlation - Audit trail or timestamp tracking",
        "VERSION_TRACKING": "Sequential with temporal patterns - Version control or change tracking",

        # ==================== CATEGORY 6: WEAK/UNKNOWN (3) ====================
        "WEAK_CORRELATION": "Some statistical correlation without clear semantic pattern",
        "INDEPENDENT_COLUMNS": "No clear relationship detected - Likely unrelated columns",
        "AMBIGUOUS_RELATIONSHIP": "Conflicting semantic signals - Requires manual review"
        }

# ============================================================================
# NODE EMBEDDING GENERATION
# ============================================================================

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
        sample_size=int(inner_content['sample_size'])
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
            full_embedding=np.concatenate(embeddings)
        else:
            full_embedding=np.zeros(8)
        if len(full_embedding) < self.feature_dim:
            padding=np.zeros(self.feature_dim - len(full_embedding))
            return np.concatenate([full_embedding, padding])
        else:
            return full_embedding[:self.feature_dim]
    def _engineer_metadata_features(self, dtype, sample_size, content_text):
        features=[
            1.0 if 'int' in dtype else 0.0,
            1.0 if 'float' in dtype else 0.0,
            1.0 if 'object' in dtype else 0.0,
            sample_size/100.0,
            content_text.count('<NULL>')/sample_size,
            len(content_text)/1000.0,
            content_text.count('|')/sample_size,
            1.0 if any(c.isdigit() for c in content_text) else 0.0,
        ]
        return np.array(features)

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class GraphBuilder:
    """
    IMPORTANT: This is the same GraphBuilder from table2graph_sem.py, but
    modified for contrastive learning. In the original version,
    _create_supervised_edges returns edge_labels (class indices). For contrastive
    learning, we need to store edge_features (the 10 computed features) instead.

    Question for user: Should I modify _create_supervised_edges now to return
    edge_features instead of edge_labels, or keep it as-is for now?
    """
    def __init__(self, content_extractor, feature_tokenizer, relationship_generator, semantic_label_generator=None, mode='train'):
        self.content_extractor=content_extractor
        self.feature_tokenizer=feature_tokenizer
        self.relationship_generator=relationship_generator
        self.semantic_label_generator=semantic_label_generator
        self.mode=mode

    def build_graph(self, df):
        """Main orchestration method - returns torch_geometric Data object"""
        node_features, node_mapping=self._create_embedded_nodes(df)
        if self.mode=='train':
            # For contrastive learning, we only need edges for sparse graph construction
            edge_index=self._create_sparse_edges(df, node_mapping)
        else:
            edge_index=self._create_candidate_edges(df, node_mapping)
        return self._to_pytorch_geometric(node_features, edge_index)

    def _create_embedded_nodes(self, df):
        """Creates 512-d node embeddings for each column"""
        node_features=[]
        node_mapping={}
        # Fit TF-IDF vectorizer if needed
        if self.feature_tokenizer.vectorizer and not hasattr(self.feature_tokenizer.vectorizer, 'vocabulary_'):
            all_content=[]
            for col in df.columns:
                content_dict=self.content_extractor.get_col_stats(df, col)
                inner_content=list(content_dict.values())[0]
                all_content.append(inner_content['column_content'])
            self.feature_tokenizer.vectorizer.fit(all_content)
        # Encode each column
        for idx, col in enumerate(df.columns):
            content_dict=self.content_extractor.get_col_stats(df, col)
            embedding=self.feature_tokenizer.encode_column_content(content_dict)
            node_features.append(embedding)
            node_mapping[col]=idx
        return torch.stack([torch.tensor(f, dtype=torch.float32) for f in node_features]), node_mapping

    def _create_sparse_edges(self, df, node_mapping):
        """Creates sparse graph based on composite threshold filtering"""
        relationships=self.relationship_generator.compute_all_relationship_scores(df)
        edge_index=[]
        for rel in relationships:
            # Only add edges above composite threshold
            if rel.get('composite_score', 0)>=self.relationship_generator.thresholds['composite_threshold']:
                src_idx=node_mapping[rel['col1']]
                dst_idx=node_mapping[rel['col2']]
                # Undirected edges
                edge_index.extend([[src_idx, dst_idx], [dst_idx, src_idx]])
        if not edge_index:
            return torch.empty((2,0), dtype=torch.long)
        return torch.tensor(edge_index).T

    def _create_candidate_edges(self, df, node_mapping):
        """Creates candidate edges for test/inference"""
        relationships=self.relationship_generator.compute_all_relationship_scores(df)
        edge_index=[]
        for rel in relationships:
            src_idx=node_mapping[rel['col1']]
            dst_idx=node_mapping[rel['col2']]
            edge_index.extend([[src_idx, dst_idx], [dst_idx, src_idx]])
        return torch.tensor(edge_index).T if edge_index else torch.empty((2,0), dtype=torch.long)

    def _to_pytorch_geometric(self, node_features, edge_index):
        """Converts to PyTorch Geometric Data object"""
        return Data(x=node_features, edge_index=edge_index)

class QuestionEncoder(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2', freeze=True):
        super().__init__()
        self.encoder=SentenceTransformer(model_name)
        self.output_dim=self.encoder.get_sentence_embedding_dimension()
        self.freeze=freeze
        #Freeze Encoder weights if specified
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad=False
    def forward(self, questions):
        context=torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            embeddings=self.encoder.encode(
                questions,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return embeddings.float()
    def encode(self, questions):
        return self.forward(questions)

# ============================================================================
# CONTRASTIVE GNN ENCODER
# ============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        #Attention MLP: input_dim -> input_dim//2 -> 1
        self.attention_mlp=nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.Tanh(),
            nn.Linear(input_dim//2, 1)
        )
    def forward(self, node_embeddings, batch=None):
        #Compute attention scores for each node
        attention_scores=self.attention_mlp(node_embeddings)
        if batch is None:
            #Single Graph: Softmax over all nodes
            attention_weights=F.softmax(attention_scores, dim=0)
            graph_embedding=(attention_weights*node_embeddings).sum(dim=0, keepdim=True)
        else:
            #Batched Graphs
            from torch_geometric.nn import global_add_pool
            #Initialize weights
            attention_weights=torch.zeros_like(attention_scores)
            #Apply softmax per graph
            for graph_id in batch.unique():
                mask=(batch==graph_id)
                attention_weights[mask]=F.softmax(attention_scores[mask], dim=0)
            #Weighted sum per graph
            weighted_nodes=attention_weights*node_embeddings
            graph_embedding=global_add_pool(weighted_nodes, batch)
        return graph_embedding

class ContrastiveGNNEncoder(nn.Module):
    """
    Encodes table graphs into embeddings aligned with question space.
    
    Architecture:
        Table → GNN (2-layer) → Attention Pooling → Projection Head → 768-d embedding
    
    Design rationale:
        - 2-layer GNN: Captures compositional patterns without over-smoothing
        - Attention pooling: Learns which columns matter for each question
        - Projection head: Maps graph space (256-d) to question space (768-d)
        - L2 normalization: Enables cosine similarity for contrastive loss
    """
    def __init__(self, node_dim=512, hidden_dim=256, output_dim=768, num_layers=2):
        super().__init__()
        #Component 1: GNN for message passing(resuse TableGCN from gcn_conv.py)
        self.GNN=TableGCN(
            input_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1
        )
        #Component 2: Attention Pooling for graph-level representation
        self.attention_pool=AttentionPooling(hidden_dim)
        #Component 3: Projection head to question space
        #Gradual Expansion 256 -> 512 -> 768
        self.projection_head=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, output_dim)
        )
    def forward(self, pyg_data, batch=None):
        """
        Encode table graph into dense embedding.
        
        Args:
            pyg_data: PyTorch Geometric Data object
                     - x: [num_nodes, node_dim] node features
                     - edge_index: [2, num_edges] edge connectivity
            batch: [num_nodes] - batch assignment (None for single graph)
        
        Returns:
            graph_embedding: [1, output_dim] if batch=None, else [batch_size, output_dim]
                            L2-normalized for cosine similarity
        """
        #Step 1: GNN Message Passing
        node_embeddings=self.GNN(pyg_data.x, pyg_data.edge_index, batch)
        #Step 2: Attention based pooling to graph level representation
        graph_embedding=self.attention_pool(node_embeddings, batch)
        #Step 3: Project to question space
        projected_embedding=self.projection_head(graph_embedding)
        #Step 4: Normalize (L2) for cosine similarity, enables dot product sim
        projected_embedding=F.normalize(projected_embedding, p=2, dim=-1)
        return projected_embedding


# ============================================================================
# CONTRASTIVE LOSS
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise-Contrastive Estimation) Loss.
    Used for contrastive learning with in-batch negatives.
    
    Loss encourages:
    - High similarity between positive pairs (table, matching question)
    - Low similarity between negative pairs (table, non-matching questions)
    
    Mathematical formulation:
        For each table_i in batch:
        Loss_i = -log(exp(sim(table_i, question_i) / τ) / Σ_j exp(sim(table_i, question_j) / τ))
    
    Where:
        - sim(a, b) = cosine similarity (dot product for L2-normalized embeddings)
        - τ = temperature parameter (controls distribution sharpness)
        - j ranges over all questions in batch (in-batch negatives)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature=temperature
        self.criterion=nn.CrossEntropyLoss()
    def forward(self, table_embeddings, question_embeddings, labels=None):
        batch_size=table_embeddings.size(0)
        #Default: diagonal matching
        if labels is None:
            labels=torch.arange(batch_size, device=table_embeddings.device)
        #Compute similarity matrix
        similarity_matrix=torch.matmul(table_embeddings,question_embeddings.T)/self.temperature
        #Apply Cross Entropy
        loss=self.criterion(similarity_matrix, labels)
        return loss
    

# ============================================================================
# QUESTION ENCODER
# ============================================================================

class QuestionGenerator:
    def __init__(self, semantic_label_generator):
        self.label_gen=semantic_label_generator
        self.templates=self._create_question_templates()
    
    def _create_question_templates(self):
        """
        Create question templates for semantic relationship types.
        Focuses on 12 most common types, 3 variations each.
        """
        templates = {
            'PRIMARY_FOREIGN_KEY': [
                "Which columns establish a primary-foreign key relationship?",
                "What are the key columns that link these entities?",
                "Which fields define the referential integrity constraint?"
            ],
            'TEMPORAL_SEQUENCE': [
                "Which columns form a temporal sequence?",
                "What time-ordered relationship exists between these fields?",
                "Which columns track chronological progression?"
            ],
            'AGGREGATION': [
                "Which columns are related through aggregation?",
                "What is the summary relationship between these fields?",
                "Which column aggregates values from the other?"
            ],
            'HIERARCHY': [
                "Which columns form a hierarchical relationship?",
                "What is the parent-child relationship here?",
                "Which fields define the organizational structure?"
            ],
            'CATEGORICAL_GROUPING': [
                "Which columns group data categorically?",
                "What categorical relationship exists between these fields?",
                "Which column categorizes the other?"
            ],
            'ONE_TO_MANY': [
                "Which columns have a one-to-many relationship?",
                "What is the cardinality relationship between these fields?",
                "Which column maps to multiple values in the other?"
            ],
            'MANY_TO_MANY': [
                "Which columns have a many-to-many relationship?",
                "What is the bidirectional mapping between these fields?",
                "Which columns form a junction relationship?"
            ],
            'DERIVED_CALCULATED': [
                "Which columns are related through calculation?",
                "What is the derived relationship between these fields?",
                "Which column is computed from the other?"
            ],
            'COMPOSITE_KEY': [
                "Which columns form a composite key?",
                "What is the multi-column key relationship?",
                "Which fields together define uniqueness?"
            ],
            'MEASUREMENT_UNIT': [
                "Which columns represent the same measurement in different units?",
                "What is the unit conversion relationship?",
                "Which fields measure the same quantity?"
            ],
            'VERSIONING': [
                "Which columns track versioning information?",
                "What is the version history relationship?",
                "Which fields manage record versions?"
            ],
            'STATUS_TRANSITION': [
                "Which columns track status transitions?",
                "What is the state change relationship?",
                "Which fields define workflow progression?"
            ]
        }
        return templates
    def generate_questions_for_table(self, df, relationships, num_positive=10, num_negative=10):
        questions=[]
        columns=list(df.columns)
        #-------------------- POSITIVE QUESTIONS --------------------------
        if len(relationships)>0:
            #Sample Subset of relationships
            sampled_rels=np.random.choice(
                relationships,
                size=min(num_positive, len(relationships)),
                replace=False
            ).tolist()
            for rel in sampled_rels:
                #Get Semantic Label for this relationship
                semantic_label=rel.get('semantic_label', np.random.choice(list(self.templates.keys())))
                #Get template for this label
                if semantic_label in self.templates:
                    question_template=np.random.choice(self.templates[semantic_label])
                else:
                    #Fallback for unsupported labels
                    question_template="What is the relationship between these columns?"
                #Add Column Names to question
                col1=rel['col1']
                col2=rel['col2']
                question=f"{question_template} Focus on '{col1}' and '{col2}'."
                questions.append({
                    'table': df,
                    'question':question,
                    'label': 1, #Positive
                    'columns': [col1, col2],
                    'semantic_label':semantic_label
                })
        #---------------------NEGATIVE QUESTIONS------------------------
        #Build set of existing relationship pairs
        relationship_pairs={(rel['col1'], rel['col2']) for rel in relationships}
        relationship_pairs.update({(rel['col2'], rel['col1']) for rel in relationships})
        negative_count=0
        max_attempts=num_negative*10
        attempts=0
        while negative_count<num_negative and attempts<max_attempts:
            attempts+=1
            #Sample random column pair
            if len(columns)<2:
                break
            col1,col2=np.random.choice(columns, size=2, replace=False)
            #Check if this pair is NOT in relationships (hard negatives)
            if (col1, col2) not in relationship_pairs:
                #Pick random semantic label
                semantic_label=np.random.choice(list(self.templates.keys()))
                question_template=np.random.choice(self.templates[semantic_label])

                question=f"{question_template} Focus on '{col1}' and '{col2}'."
                questions.append({
                    'table':df,
                    'question':question,
                    'label':0,
                    'columns':[col1, col2],
                    'semantic_label':semantic_label
                })
                negative_count+=1
        return questions
    def generate_dataset(self, tables, relationship_generator, num_per_table=20):
        #Generate full contrastive dataset across multiple tables
        all_questions=[]
        num_positive=num_per_table//2
        num_negative=num_per_table-num_positive
        for df in tables:
            try:
                #Generate relationships
                relationships=relationship_generator.generate_relationships(df)
                #Generate questions
                table_questions=self.generate_questions_for_table(
                    df,
                    relationships,
                    num_positive=num_positive,
                    num_negative=num_negative
                )
                all_questions.extend(table_questions)
            except Exception as e:
                print(f"Warning: Failed to generate questiosn for this table: {e}")
                continue
        return all_questions


# ============================================================================
# PHASE 5: TABLE-QUESTION DATASET
# ============================================================================

class TableQuestionDataset(Dataset):
    """
    PyTorch Dataset for table-question pairs in contrastive learning.
    Converts tables to PyG graphs on-the-fly.
    """
    def __init__(self, question_data, data_processor, pyg_converter):
        """
        Args:
            question_data: list of dicts from QuestionGenerator.generate_dataset()
                Format: [{'table': df, 'question': str, 'label': int, ...}, ...]
            data_processor: DataProcessor instance
            pyg_converter: PyGConverter instance (converts table → PyG graph)
        """
        self.question_data = question_data
        self.data_processor = data_processor
        self.pyg_converter = pyg_converter

    def __len__(self):
        return len(self.question_data)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'graph': PyG Data object,
                'question': str,
                'label': int (0 or 1)
            }
        """
        item = self.question_data[idx]
        df = item['table']
        question = item['question']
        label = item['label']

        # Convert table to PyG graph using existing pipeline
        pyg_data = self.pyg_converter.convert_table(df)

        return {
            'graph': pyg_data,
            'question': question,
            'label': label
        }


def collate_fn(batch):
    """
    Custom collate function for batching graphs with questions.

    Args:
        batch: list of dicts from TableQuestionDataset.__getitem__

    Returns:
        batched_graphs: PyG Batch object (batched graphs with global batch index)
        questions: list of question strings
        labels: torch.LongTensor of shape [batch_size]
    """
    graphs = [item['graph'] for item in batch]
    questions = [item['question'] for item in batch]
    labels = torch.LongTensor([item['label'] for item in batch])

    # Batch graphs using PyG's Batch.from_data_list
    # This automatically creates a 'batch' attribute for node-to-graph mapping
    batched_graphs = Batch.from_data_list(graphs)

    return batched_graphs, questions, labels


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Convenience function to create DataLoader with custom collate.

    Args:
        dataset: TableQuestionDataset instance
        batch_size: number of table-question pairs per batch
        shuffle: whether to shuffle data
        num_workers: number of subprocesses for data loading

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


# ============================================================================
# CONTRASTIVE TRAINER
# ============================================================================

class ContrastiveTrainer:
    """
    Training loop for contrastive table-question learning.
    Optimizes InfoNCE loss to align graph and question embeddings.
    """
    def __init__(
            self,
            graph_encoder,
            question_encoder,
            loss_fn,
            learning_rate=1e-4):
        self.graph_encoder=graph_encoder
        self.question_encoder=question_encoder
        self.loss_fn=loss_fn
        #Only optimize graph encoder (question encoder is frozen)
        self.optimizer=torch.optim.AdamW(
            self.graph_encoder.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        self.train_losses=[]
        self.val_losses=[]
    def train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader from create_dataloader()
        
        Returns:
            float: average loss for the epoch
        """
        self.graph_encoder.train()
        epoch_loss=0.0
        num_batches=0
        for batched_graphs, questions, labels in dataloader:
            #Forward Pass
            graph_embeddings=self.graph_encoder(
                batched_graphs,
                batch=batched_graphs.batch
            )
            question_embeddings=self.question_encoder(questions)
            #Compute loss
            loss=self.loss_fn(graph_embeddings, question_embeddings, labels)
            #Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.graph_encoder.parameters(),max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss+=loss.item()
            num_batches+=1
        avg_loss=epoch_loss/num_batches if num_batches>0 else 0.0
        return avg_loss
    @torch.no_grad()
    def validate(self, dataloader):
        self.graph_encoder.eval()
        total_loss=0.0
        num_batches=0

        all_graph_embeddings=[]
        all_question_embeddings=[]
        all_labels=[]
        for batched_graphs, questions, labels in dataloader:
            #Forward Pass
            graph_embeddings=self.graph_encoder(
                batched_graphs,
                batch=batched_graphs.batch
            )
            question_embeddings=self.question_encoder(questions)
            #Compute loss
            loss=self.loss_fn(graph_embeddings, question_embeddings, labels)
            total_loss+=loss.item()
            num_batches+=1
            #Store for recall calculations
            all_graph_embeddings.append(graph_embeddings)
            all_question_embeddings.append(question_embeddings)
            all_labels.append(labels)
        
        avg_loss=total_loss/num_batches if num_batches>0 else 0.0

        #Compute Recall@K metrics
        graph_embs= torch.cat(all_graph_embeddings, dim=0)
        question_embs=torch.cat(all_question_embeddings, dim=0)
        labels_tensor=torch.cat(all_labels, dim=0)
        recall_1=self._compute_recall_at_k(
            graph_embs, question_embs, labels_tensor, k=1
        )
        recall_5=self._compute_recall_at_k(
            graph_embs, question_embs, labels_tensor, k=5
        )
        return {
            'loss':avg_loss,
            'recall@1':recall_1,
            'recall@5': recall_5
        }
    def _compute_recall_at_k(self, graph_embeddings, question_embeddings, labels, k=1):
        """
        Compute Recall@K for positive pairs.
        
        For each question, check if the correct graph is in top-K retrieved graphs.
        
        Args:
            graph_embeddings: [N, 768]
            question_embeddings: [N, 768]
            labels: [N] (1 for positive, 0 for negative)
            k: top-k to consider
        
        Returns:
            float: recall@k (percentage of positive pairs retrieved in top-k)
        """
        # Only evaluate on postive pairs
        positive_mask=labels==1
        if positive_mask.sum()==0:
            return 0.0
        #Compute Similarity Matrix: [N_questions, N_graphs]
        similarity=torch.matmul(question_embeddings, graph_embeddings.T)
        #For each question, get top-k 'graphs?'
        _, top_k_indices=torch.topk(similarity, k=min(k, similarity.size(1)),dim=1)
        #Check if correct graph is in top-k for positive pairs
        correct_indices=torch.arange(len(labels), device=labels.device)
        recalls=[]
        for i in range(len(labels)):
            if labels[i]==1: #Only check +ve pairs
                #Is the correct graph (index i) in top k questions
                is_in_top_k=correct_indices[i] in top_k_indices[i]
                recalls.append(is_in_top_k.float().item())
        recall_at_k=sum(recalls)/len(recalls) if recalls else 0.0
        return recall_at_k
    def train(self, train_loader, val_loader, num_epochs=10, print_every=1):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: number of epochs to train
            print_every: print metrics every N epochs
        
        Returns:
            dict: training history
        """
        print(f"Starting training for {num_epochs} epochs ...")
        for epoch in range(num_epochs):
            #Train
            train_loss=self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            #Validate
            val_metrics=self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            if (epoch+1)%print_every==0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Recall@1: {val_metrics['recall@1']:.3f} | "
                f"Recall@5: {val_metrics['recall@5']:.3f} | ")
        print("Training Complete")
        return {
            'train_losses':self.train_losses,
            'val_losses': self.val_losses
        }
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'graph_encoder_state':self.graph_encoder.state_dict(),
            'optimizer_state':self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses':self.val_losses
        }, path)
        print(f"Checkpoint saved to {path}")
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint=torch.load(path)
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_losses=checkpoint.get('train_losses', [])
        self.val_losses=checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {path}")


# ============================================================================
# TODO: COMPONENTS TO BE ADDED NEXT (ALL DONE)
# ============================================================================

"""
The following components need to be implemented for contrastive learning:

1. QuestionEncoder (~50 lines)
   - Encodes natural language questions into 768-d embeddings
   - Uses sentence-transformers or similar
   - Input: question string
   - Output: 768-d tensor

2. ContrastiveGNNEncoder (~150 lines)
   - TableGCN for node embedding enrichment (reuse from gcn_conv.py)
   - Attention pooling for graph-level embeddings
   - Projection head to map to question space (512->768)
   - Input: PyG Data object
   - Output: 768-d graph embedding

3. InfoNCELoss (~30 lines)
   - Contrastive loss implementation
   - Handles in-batch negatives
   - Temperature scaling parameter
   - Input: graph embeddings, question embeddings, labels
   - Output: scalar loss

4. QuestionGenerator (~200 lines)
   - Generates synthetic questions from table metadata
   - Uses 34 semantic relationship types as templates
   - Creates hard negatives
   - Input: DataFrame, relationships
   - Output: list of (table, question, label) tuples

5. ContrastiveTrainer (~150 lines)
   - Training loop for contrastive learning
   - Batch construction with hard negatives
   - Evaluation metrics (Recall@K)
   - Input: dataset, hyperparams
   - Output: trained model

6. TableQuestionDataset (~100 lines)
   - PyTorch Dataset wrapper
   - Handles table-question pairs
   - Batch collation
   - Input: list of (df, question, label)
   - Output: batched PyG Data + question tensors

Total new code: ~680 lines
Total reused code: ~1200 lines (60-70% reuse rate)
"""
