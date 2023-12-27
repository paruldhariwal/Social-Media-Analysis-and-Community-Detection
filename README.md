This project focuses on analyzing social media network structures using Python and the igraph library. The goal is to extract meaningful insights from the network data, understand user interactions, and predict connections within the network.

Overview
The analysis comprises several key components:

1. Data Preprocessing
Dataset Loading: The project involves loading social media network data (in .edges format) into the igraph graph structure.
2. Link Prediction
Feature Crafting: Crafted intricate feature vectors based on node attributes and network characteristics.
Models Used: Utilized Ridge Regression and Support Vector Machine (SVM) models for precise link prediction.
Evaluation: Assessed the accuracy of link existence forecasts using metrics like log loss.
3. Community Detection
Louvain Method: Applied the Louvain method for community detection, revealing cohesive clusters of users within the network.
4. Centrality Measures
Identifying Influential Nodes: Conducted centrality measures such as Eigenvector, Closeness, and Betweenness to identify key nodes in the network.
Visualization: Employed graph plotting and visualization techniques to visualize the network's centrality measures.
5. 3D Visualization
Interactive Node Representations: Leveraged data visualization techniques using Plotly to create immersive 3D representations of Facebook nodes.
Insight Generation: These visualizations enhanced the understanding of network dynamics and interconnections.
