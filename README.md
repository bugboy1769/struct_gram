# Problem Statement: Text and Structured Data are two separate modalities in semantic terms. There is crossover but the connection structure and grammar is different.

## Central Approach: 
    
* Convert structured data into graphs using NetworkX graphs, for now [Columns, Column Attributes -> Node Features; Correlations, Relational Information -> Edges Features]. Tokenize text attributes using huggingface AutoTokenizer.
* Use a simple GNN with the simplest aggregation function to inform nodes and edges of neighbourhood. Current graph state then reflects neighbourhood knowledge.
* Pass the GNN output (maybe some super tokens: S_i) through a trainable projection layer producing S_o, and then append the output at the start of an LLMs input query, Q. Giving, S_o:Q.
* Training Data will be a one to many mapping, the change will be in the query being appended and the answer that is produced (O). S_o:Q_x -> O_x. Where Q_x and O_x are the sets of text inputs and outputs.

## Current State: A graph exists, features exist. GNN has been initialised along with forward pass. Major effort is feature construction, if I have more time I would like to automate it using LLMs.

ToDo: Finalise Features and feature vector encoding method. Setup projection layer.

Note: The one to many mapping is quite reductive, a much better approach would be to have different super tokens using a variety of {feature_vectors:aggregation_method} combinations. Also note, the only trainable object here for now is the projection layer and this is not graph querying, its passive context creation attempting to capture structured data semantics.

Getting to GCN:
Text stats → Tokens → Token embeddings → Pool → Node features → GCN → Node embeddings → Pool → Graph embedding → MLP → LLM
           (many tokens)              (1 per node)        (1 per node)           (1 for graph)

The current graph state looks something like this:

<img width="640" height="480" alt="good_graph" src="https://github.com/user-attachments/assets/5d6b5f17-a8a8-484d-b1b3-f5a853f25709" />

