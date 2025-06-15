import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')
from D2FG import DF2G

class GraphVisualizer:
    """
    Comprehensive NetworkX graph visualization toolkit.
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.pos = None  # Will store node positions
        
    def basic_matplotlib_viz(self, figsize=(12, 8), node_color_attr='node_type', 
                           edge_color_attr='edge_type', save_path=None):
        """
        Basic matplotlib visualization with colored nodes and edges.
        """
        plt.figure(figsize=figsize)
        
        # Calculate layout
        if self.pos is None:
            self.pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Get node colors based on node type
        node_types = [self.graph.nodes[node].get(node_color_attr, 'unknown') 
                     for node in self.graph.nodes()]
        unique_node_types = list(set(node_types))
        node_color_map = dict(zip(unique_node_types, 
                                plt.cm.Set3(np.linspace(0, 1, len(unique_node_types)))))
        node_colors = [node_color_map[nt] for nt in node_types]
        
        # Get edge colors based on edge type
        edge_types = [self.graph.edges[edge].get(edge_color_attr, 'unknown') 
                     for edge in self.graph.edges()]
        unique_edge_types = list(set(edge_types))
        edge_color_map = dict(zip(unique_edge_types, 
                                plt.cm.Set1(np.linspace(0, 1, len(unique_edge_types)))))
        edge_colors = [edge_color_map[et] for et in edge_types]
        
        # Draw the graph
        nx.draw_networkx_nodes(self.graph, self.pos, 
                              node_color=node_colors, 
                              node_size=300, 
                              alpha=0.8)
        
        nx.draw_networkx_edges(self.graph, self.pos, 
                              edge_color=edge_colors, 
                              alpha=0.6, 
                              width=2)
        
        # Add labels for smaller graphs
        if len(self.graph.nodes()) < 50:
            labels = {node: str(node)[:10] + '...' if len(str(node)) > 10 else str(node) 
                     for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, self.pos, labels, font_size=8)
        
        # Create legends
        self._create_matplotlib_legend(node_color_map, edge_color_map, 
                                     node_color_attr, edge_color_attr)
        
        plt.title(f"Graph Visualization ({len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges)")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def hierarchical_viz(self, figsize=(15, 10), save_path=None):
        """
        Hierarchical visualization separating different node types into layers.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group nodes by type
        node_groups = {}
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get('node_type', 'unknown')
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append(node)
        
        # Create hierarchical positions
        pos = {}
        colors = plt.cm.Set3(np.linspace(0, 1, len(node_groups)))
        node_colors = {}
        
        y_offset = 0
        layer_height = 2
        
        for i, (node_type, nodes) in enumerate(node_groups.items()):
            # Arrange nodes in this layer
            if len(nodes) == 1:
                x_positions = [0]
            else:
                x_positions = np.linspace(-len(nodes)/2, len(nodes)/2, len(nodes))
            
            for j, node in enumerate(nodes):
                pos[node] = (x_positions[j], y_offset)
                node_colors[node] = colors[i]
            
            y_offset += layer_height
        
        # Draw nodes by type
        for node_type, nodes in node_groups.items():
            node_pos = {n: pos[n] for n in nodes}
            nx.draw_networkx_nodes(self.graph, node_pos, 
                                  nodelist=nodes,
                                  node_color=[node_colors[n] for n in nodes],
                                  node_size=500, 
                                  alpha=0.8,
                                  label=node_type)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1)
        
        # Add labels
        if len(self.graph.nodes()) < 30:
            labels = {node: str(node)[:15] + '...' if len(str(node)) > 15 else str(node) 
                     for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Hierarchical Graph Layout")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        return pos
    
    def interactive_plotly_viz(self, save_path=None):
        """
        Interactive Plotly visualization with hover information.
        """
        # Calculate layout
        if self.pos is None:
            self.pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        
        # Color mapping for node types
        node_types = [self.graph.nodes[node].get('node_type', 'unknown') 
                     for node in self.graph.nodes()]
        unique_types = list(set(node_types))
        color_discrete_map = dict(zip(unique_types, px.colors.qualitative.Set3[:len(unique_types)]))
        
        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info for hover
            attrs = self.graph.nodes[node]
            node_type = attrs.get('node_type', 'unknown')
            
            # Create hover text
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {node_type}<br>"
            
            # Add relevant attributes
            for key, value in attrs.items():
                if key != 'node_type' and len(str(value)) < 100:
                    hover_text += f"{key}: {value}<br>"
            
            node_info.append(hover_text)
            node_text.append(str(node)[:20])
            node_colors.append(color_discrete_map[node_type])
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge info
            attrs = self.graph.edges[edge]
            edge_type = attrs.get('edge_type', 'unknown')
            edge_info.append(f"Edge: {edge[0]} → {edge[1]}<br>Type: {edge_type}")
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=1, color='rgba(125,125,125,0.5)'),
                                hoverinfo='none',
                                mode='lines',
                                name='Edges'))
        
        # Add nodes
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                marker=dict(size=15,
                                          color=node_colors,
                                          line=dict(width=2, color='white')),
                                text=node_text,
                                textposition="middle center",
                                hovertext=node_info,
                                hoverinfo='text',
                                name='Nodes'))
        
        fig.update_layout(title=f"Interactive Graph Visualization<br>{len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges",
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[ dict(
                             text="Hover over nodes and edges for details",
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002,
                             xanchor='left', yanchor='bottom',
                             font=dict(color='gray', size=12)
                         )],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         plot_bgcolor='white')
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    
    def subgraph_viz(self, node_type_filter=None, max_nodes=50, figsize=(12, 8)):
        """
        Visualize a filtered subgraph based on node types or size limits.
        """
        if node_type_filter:
            # Filter nodes by type
            filtered_nodes = [node for node, attrs in self.graph.nodes(data=True)
                            if attrs.get('node_type') in node_type_filter]
        else:
            # Just limit size
            filtered_nodes = list(self.graph.nodes())[:max_nodes]
        
        subgraph = self.graph.subgraph(filtered_nodes)
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
        
        # Color by node type
        node_types = [subgraph.nodes[node].get('node_type', 'unknown') 
                     for node in subgraph.nodes()]
        
        # Draw
        nx.draw(subgraph, pos, 
                with_labels=True, 
                node_color=plt.cm.Set3(np.linspace(0, 1, len(set(node_types)))),
                node_size=500,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.8)
        
        plt.title(f"Subgraph Visualization\nFilter: {node_type_filter if node_type_filter else 'Size limited'}")
        plt.tight_layout()
        plt.show()
        
        return subgraph
    
    def adjacency_matrix_viz(self, figsize=(10, 8), save_path=None):
        """
        Visualize the graph as an adjacency matrix heatmap.
        """
        # Get adjacency matrix
        adj_matrix = nx.adjacency_matrix(self.graph).todense()
        nodes = list(self.graph.nodes())
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(adj_matrix, 
                   xticklabels=[str(n)[:10] for n in nodes],
                   yticklabels=[str(n)[:10] for n in nodes],
                   cmap='Blues',
                   cbar_kws={'label': 'Connection'})
        
        plt.title("Graph Adjacency Matrix")
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def _create_matplotlib_legend(self, node_color_map, edge_color_map, 
                                node_attr, edge_attr):
        """Create legends for matplotlib plots."""
        # Node type legend
        node_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=color, markersize=10,
                                         label=node_type) 
                              for node_type, color in node_color_map.items()]
        
        # Edge type legend  
        edge_legend_elements = [plt.Line2D([0], [0], color=color, linewidth=3,
                                         label=edge_type)
                              for edge_type, color in edge_color_map.items()]
        
        # Add legends
        first_legend = plt.legend(handles=node_legend_elements, 
                                title=f"Node Types ({node_attr})",
                                loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.gca().add_artist(first_legend)
        
        plt.legend(handles=edge_legend_elements, 
                  title=f"Edge Types ({edge_attr})",
                  loc='upper left', bbox_to_anchor=(1, 0.5))

# Usage example with your DataFrame to Graph converter
def demonstrate_visualizations():
    """
    Complete example showing how to visualize graphs created from DataFrames.
    """
    # Create sample data (same as before)
    sample_data = {
        'employee_id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing'],
        'salary': [75000, 65000, 80000, 60000, 70000],
        'years_experience': [3, 5, 7, 2, 4],
        'location': ['NYC', 'SF', 'NYC', 'LA', 'SF']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Assuming you have the DataFrameToGraph class from the previous artifact
    # converter = DataFrameToGraph(df, strategy="hybrid")
    # graph = converter.create_graph()
    
    # For demo purposes, create a simple graph
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (1, 3)])
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['node_type'] = ['entity', 'attribute', 'value'][i % 3]
    
    for edge in G.edges():
        G.edges[edge]['edge_type'] = 'connection'
    
    # Create visualizer
    viz = GraphVisualizer(G)
    
    print("1. Basic Matplotlib Visualization:")
    viz.basic_matplotlib_viz()
    
    print("\n2. Hierarchical Layout:")
    viz.hierarchical_viz()
    
    print("\n3. Interactive Plotly Visualization:")
    viz.interactive_plotly_viz()
    
    print("\n4. Adjacency Matrix:")
    viz.adjacency_matrix_viz()
    
    return viz

if __name__ == "__main__":
    demonstrate_visualizations()