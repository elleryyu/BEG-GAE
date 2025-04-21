import pickle
from data import Graph_with_knn  

if __name__ == '__main__':
    # 
    embedding_path = '../saved_pkl/graph_embeddings_site_16_1021.pkl'  # 
    label_path = '../site_16/site_16_labels.csv'  # 

    dataset = Graph_with_knn(embedding_path=embedding_path, label_path=label_path)


    graph, node_ids = dataset[0]


    with open('graph_knn.pkl', 'wb') as f:
        pickle.dump(graph, f)  # 保存图
        pickle.dump(node_ids, f)  # 保存节点ID

    print("Graph and node IDs saved to 'graph_knn.pkl'.")