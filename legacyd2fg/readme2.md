#Renewed Architectural Approach Towards GNN - LLM Interfacing

	## Key Changes:

		* Number of Graphs per table: The number of graphs per table is going to be greater than 1. This is to solve the edge relationality problem. Now, we will create links for data types. Connect the columns that have similar data types. And then one graph will be the inverse, connections for disparate datatypes.
		* Embedding Methodology: We will not use LLM Tokenizers anymore. We will instead use SBERT to embed the node and edge features for every graph. We will then project these embeddings to a smaller dimensional space for efficient GNN computation.
		* Training Paradigm Upto GNN: We will perform missing link training on the GNNs. If I am correct, the GNN projection layer does not need to be trained as such, the link relationality will get captured by the GNN anyway. The only concern is when the GNN output gets reprojected into an LLMs native space, will the earlier projection cause huge semantic flow problems. (Need to search.)
		* Projection Layer Problem: Open.
