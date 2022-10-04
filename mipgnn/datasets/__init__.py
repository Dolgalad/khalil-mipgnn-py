import os
import torch
import networkx as nx

from progress.bar import Bar

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.loader import DataLoader

from mipgnn.predict import create_data_object

def list_processed_datasets():
    return list(os.listdir(os.path.expanduser("~/.cache/mipgnn/datasets/processed")))

# Preprocessing to create Torch dataset.
class GraphDataset(InMemoryDataset):

    def __init__(self, name, root, data_path, bias_threshold, transform=None, pre_transform=None,
                 pre_filter=None):

        self.bias_threshold = bias_threshold
        self.name = name
        self.data_path = data_path

        super(GraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return self.name

    @property
    def processed_file_names(self):
        return self.name

    def download(self):
        pass

    def process(self):
        #print(f"Preprocessing files in {self.data_path}.")

        data_list = []

        graph_files = [f for f in os.listdir(self.data_path) if f.endswith("graph_bias.pkl")]
        num_graphs = len(graph_files)

        with Bar(f"Processing {self.name}: ", max=num_graphs) as bar:
            # Iterate over instance files and create data objects.
            for num, filename in enumerate(graph_files):
                # Get graph.
                graph = nx.read_gpickle(os.path.join(self.data_path, filename))
                # get data object
                data,_,_ = create_data_object(graph, self.bias_threshold)
                data_list.append(data)
                bar.next()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        #print("Data saved to ", self.processed_paths)
        #a = input("")


# Preprocess indices of bipartite graphs to make batching work.
class MyData(Data):
    def __inc__(self, key, value, store):
        if key in ['edge_index_var']:
            return torch.tensor([self.num_nodes_var, self.num_nodes_con]).view(2, 1)
        elif key in ['edge_index_con']:
            return torch.tensor([self.num_nodes_con, self.num_nodes_var]).view(2, 1)
        elif key in ['index']:
            return self.num_nodes_con.clone().detach()
            return torch.tensor(self.num_nodes_con)
        elif key in ['index_var']:
            return self.num_nodes_var.clone().detach()
            return torch.tensor(self.num_nodes_var)
        else:
            return 0

class MyTransform(object):
    def __call__(self, data):
        new_data = MyData()
        for key, item in data:
            new_data[key] = item
        return new_data


