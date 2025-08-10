from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import torch.nn as nn
import torch
import numpy as np


class EmbeddingToLogits(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, embeddings):
        # embeddings: [num_nodes, embedding_dim]
        logits = self.projection(embeddings)  # [num_nodes, vocab_size]
        token_ids = torch.argmax(logits, dim=-1)  # [num_nodes]
        return token_ids

def map_embeddings_to_tokens(embeddings, tokenizer):
    vocab_size = tokenizer.vocab_size
    embedding_dim = embeddings.shape[1]
    
    mapper = EmbeddingToLogits(embedding_dim, vocab_size)
    token_ids = mapper(embeddings)
    
    return token_ids.tolist()

class T5Decoder(nn.Module):

    def __init__(self, projection_layer_output_dim = 32, t5_model_name = 't5-small'):
        super().__init__()
        self.T5Model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        self.embedding_to_features = nn.Linear(projection_layer_output_dim, 8)
        self.logit_creator = EmbeddingToLogits(8, self.tokenizer.vocab_size) #Fix fix fix

    def embedding_to_prompt(self, embeds):
        features = self.embedding_to_features(embeds)
        prompts = []
        for feat in features:
            decoder_prompt = feat.detach().cpu().numpy().round(3).tolist()
            prompts.append(decoder_prompt)
        return prompts

    def forward(self, embeds, max_length = 100):
        prompts = self.embedding_to_prompt(embeds)

        generated_texts = []
        for prompt in prompts:
            # inputs = self.tokenizer(prompt, return_tensors = 'pt', padding = True, truncation = True)
            # with torch.no_grad():
            #     outputs = self.T5Model.generate(**inputs, max_length = max_length, num_beams = 4, early_stopping = True, do_sample = True, temperature = 0.7)
            generated_text = self.tokenizer.decode(prompt, skip_special_tokens =True)
            generated_texts.append(generated_text)
        return generated_texts

class DirectTensorDecoder(nn.Module):
    """
    Simpler approach: directly map tensor patterns to text descriptions
    """
    
    def __init__(self, gcn_embedding_dim=32):
        super().__init__()
        self.embedding_dim = gcn_embedding_dim
        
    def forward(self, embeds):
        descriptions = []
        
        for i, embedding in enumerate(embeds):
            values = embedding.detach().cpu().numpy()
            
            # Direct interpretation of tensor values
            desc = f"Embedding {i}: This vector has "
            
            # Describe the actual numerical pattern
            if np.mean(values) > 0.1:
                desc += "predominantly positive values, "
            elif np.mean(values) < -0.1:
                desc += "predominantly negative values, "
            else:
                desc += "balanced positive/negative values, "
                
            if np.std(values) > 0.5:
                desc += "high variance (spread out), "
            else:
                desc += "low variance (clustered), "
                
            # Find dominant dimensions
            top_indices = np.argsort(np.abs(values))[-3:]  # Top 3 dimensions
            desc += f"strongest signals in dimensions {top_indices.tolist()}, "
            
            # Describe the actual pattern
            desc += f"with peak value {np.max(values):.3f} and minimum {np.min(values):.3f}. "
            desc += f"Raw pattern: {values.round(3).tolist()[:5]}..."
            
            descriptions.append(desc)
            
        return descriptions

class SemanticMLP(nn.Module):
    def __init__(self, gcn_dim = 32, sentence_dim = 384, hidden_dim = 184):
        super().__init__()   
        self.projection = nn.Sequential(nn.Linear(gcn_dim, sentence_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(sentence_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(hidden_dim, sentence_dim),
                                        nn.LayerNorm(sentence_dim))
    def forward(self, gcn_embeddings):
        return self.projection(gcn_embeddings)
    
class EmbeddingToLogits(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, embeddings):
        # embeddings: [num_nodes, embedding_dim]
        logits = self.projection(embeddings)  # [num_nodes, vocab_size]
        token_ids = torch.argmax(logits, dim=-1)  # [num_nodes]
        return token_ids

    def map_embeddings_to_tokens(embeddings, tokenizer):
        vocab_size = tokenizer.vocab_size
        embedding_dim = embeddings.shape[1]
        
        mapper = EmbeddingToLogits(embedding_dim, vocab_size)
        token_ids = mapper(embeddings)
        
        return token_ids.tolist()

        