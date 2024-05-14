# GNN-AML
Implementing GAT with a synthetic AML dataset

Dataset is available here - https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

### **Introduction**
Money laundering remains a significant challenge in the financial sector, characterized by increasingly sophisticated methods aimed at obfuscating illicit activities. Detecting money laundering patterns is a complex task due to the intricate nature of financial transactions and the privacy constraints surrounding transactional data. In this project, we leverage Graph Neural Networks (GNNs) to detect patterns indicative of money laundering in a synthetic dataset.

### **Dataset Overview**
We utilize a synthetically generated "Anti-Money Laundering" dataset provided by IBM, created using the ALMsim simulator. The dataset comprises two groups: HI (High Illicit Ratio) and LI (Low Illicit Ratio), further divided into small, medium, and large batches. The focus is on the small and medium datasets due to computational constraints. The synthetic nature of the data allows us to explore patterns across diverse financial institutions and currencies.

### **Money Laundering Patterns**
We explore eight money laundering patterns identified by Suzumura and Kanezashi, including fan-out, fan-in, gather-scatter, scatter-gather, simple cycle, bipartite, stack, and random patterns. These patterns serve as blueprints for identifying suspicious transactional behaviors within the dataset.

### **Graphical Neural Networks (GNNs) and Message Passing Mechanism**
GNNs are a class of neural network models designed to process and represent data organized in graph structures. Here, Graph Attention Networks (GATs) are employed due to their ability to capture complex relational information within graphs. GATs utilize a message passing mechanism, where nodes exchange information with neighboring nodes, enabling comprehensive understanding of the graph structure.

### **Preprocessing and Feature Engineering**
Data preprocessing involves transforming raw transactional data into a format suitable for GNN-based modeling. Key preprocessing steps include label encoding, timestamp normalization, and feature extraction. Additionally, transactions are standardized to USD to ensure consistency across currencies.

### **Model Architecture**
The model architecture revolves around GATs, consisting of two GATConv layers followed by a linear layer with sigmoid activation. Dropout regularization is applied to prevent overfitting, and node features are aggregated to capture transactional patterns.

### **Model Training and Evaluation**
The dataset is split into training, validation, and test sets and train the model using binary cross-entropy loss and stochastic gradient descent optimizer. Model performance is evaluated on the test set, comparing predicted labels against ground truth to measure accuracy.

### **Comparative Analysis with Random Forest**
To provide context, we benchmark the GAT-based model against a random forest classifier. This comparison enables us to assess the efficacy of GNNs in detecting money laundering patterns relative to traditional machine learning approaches.
