import sys

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '.')

import matplotlib.pyplot as plt

import os
import os.path as osp
import networkx as nx
from sklearn.model_selection import train_test_split

from torchmetrics import F1Score, Precision, Recall, Accuracy

from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mipgnn.gnn_models.EdgeConv.mip_bipartite_class import SimpleNet as EdgeConv
from mipgnn.gnn_models.EdgeConv.mip_bipartite_simple_class import SimpleNet as EdgeConvSimple

from mipgnn.gnn_models.GIN.mip_bipartite_class import SimpleNet as GIN
from mipgnn.gnn_models.GIN.mip_bipartite_simple_class import SimpleNet as GINSimple

from mipgnn.gnn_models.Sage.mip_bipartite_class import SimpleNet as Sage
from mipgnn.gnn_models.Sage.mip_bipartite_simple_class import SimpleNet as SageSimple

from mipgnn.predict import create_data_object

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from mipgnn.datasets import GraphDataset, MyData, MyTransform

dataset_list = [
    #"/home/aschulz/.cache/mipgnn/datasets/test_dataset/set_cover_500_500_0.1",
    os.path.expanduser("~/.cache/mipgnn/datasets/set_cover_n1000_nv30_nc30_d0.1/train"),
    os.path.expanduser("~/.cache/mipgnn/datasets/set_cover_n1000_nv30_nc30_d0.1/test"),

]

name_list = [
    "set_cover_n1000_nv30_nc30_d0.1_train",
    "set_cover_n1000_nv30_nc30_d0.1_test",
    #"xin_set_cover_1500_1500-2000_0.1",
]

test_scores = []

for rep in [0, 1, 2, 3, 4]:
    for i in [0]:
        # Bias.
        for bias in [0.0, 0.001, 0.1]:
            # GNN.
            for m in ["ECS", "GINS", "SGS", "EC", "GIN", "SG"]:
                log = []

                # Setup model.
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                if m == "EC":
                    model = EdgeConv(hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"EC_{name_list[i]}_{bias}_{rep}"
                    print(model_name, bias, name_list[i])
                elif m == "ECS":
                    model = EdgeConvSimple(hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"ECS_{name_list[i]}_{bias}_{rep}"
                    print(model_name, bias, name_list[i])
                elif m == "GIN":
                    model = GIN(hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"GIN_{name_list[i]}_{bias}_{rep}"
                    print(model_name, bias, name_list[i])
                elif m == "GINS":
                    model = GINSimple(hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"GINS_{name_list[i]}_{bias}_{rep}"
                    print(model_name, bias, name_list[i])
                elif m == "SG":
                    model = Sage(hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"SG_{name_list[i]}_{bias}_{rep}"
                    print(model_name, bias, name_list[i])
                elif m == "SGS":
                    model = SageSimple(hidden=64, num_layers=4, aggr="mean", regression=False).to(device)
                    model_name = f"SGS_{name_list[i]}_{bias}_{rep}"
                    print(model_name, bias, name_list[i])

                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                       factor=0.8, patience=10,
                                                                       min_lr=0.0000001)

                # Prepare data.
                bias_threshold = bias
                batch_size = 10
                num_epochs = 30

                # processed dataset directory
                processed_data_root = os.path.expanduser("~/.cache/mipgnn/datasets")

                # path to non processed train data
                train_data_path = dataset_list[i]
                name_train = name_list[i]
                print(f"Train dataset {name_train}, data_path:{train_data_path}")

                # train dataset
                train_dataset = GraphDataset(name_train, processed_data_root, train_data_path, bias_threshold,
                                             transform=MyTransform()).shuffle()
                
                # test dataset
                test_data_path = dataset_list[i+1]
                name_test = name_list[i+1]
                print(f"Test dataset {name_test}, data_path:{test_data_path}")
                test_dataset = GraphDataset(name_test, processed_data_root, test_data_path, bias_threshold,
                                            transform=MyTransform()).shuffle()

                train_index, val_index = train_test_split(list(range(0, len(train_dataset))), test_size=0.2)
                #train_index, val_index = train_test_split(train_index, test_size=.2)

                val_dataset = train_dataset[val_index].shuffle()
                #test_dataset = train_dataset[test_index].shuffle() #test_dataset.shuffle()
                train_dataset = train_dataset[train_index].shuffle()

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


                def train(epoch):
                    model.train()

                    # loss_all = 0
                    zero = torch.tensor([0]).to(device)
                    one = torch.tensor([1]).to(device)

                    loss_all = 0

                    for data in train_loader:
                        data = data.to(device)

                        y = data.y_real
                        y = torch.where(y <= bias_threshold, zero, one).to(device)

                        optimizer.zero_grad()
                        output = model(data)

                        loss = F.nll_loss(output, y)
                        loss.backward()
                        loss_all += batch_size * loss.item()
                        optimizer.step()

                    return loss_all / len(train_dataset)


                @torch.no_grad()
                def test(loader):
                    model.eval()

                    zero = torch.tensor([0]).to(device)
                    one = torch.tensor([1]).to(device)
                    f1 = F1Score(num_classes=2, average="macro").to(device)
                    pr = Precision(num_classes=2, average="macro").to(device)
                    re = Recall(num_classes=2, average="macro").to(device)
                    acc = Accuracy(num_classes=2).to(device)

                    first = True
                    for data in loader:
                        data = data.to(device)
                        pred = model(data)

                        y = data.y_real

                        y = torch.where(y <= bias_threshold, zero, one).to(device)
                        pred = pred.max(dim=1)[1]

                        if not first:
                            pred_all = torch.cat([pred_all, pred])
                            y_all = torch.cat([y_all, y])
                        else:
                            pred_all = pred
                            y_all = y
                            first = False

                    return acc(pred_all, y_all), f1(pred_all, y_all), pr(pred_all, y_all), re(pred_all, y_all)


                best_val = 0.0
                test_acc = 0.0
                test_f1 = 0.0
                test_re = 0.0
                test_pr = 0.0
                for epoch in range(1, num_epochs + 1):

                    train_loss = train(epoch)
                    train_acc, train_f1, train_pr, train_re = test(train_loader)

                    val_acc, val_f1, val_pr, val_re = test(val_loader)
                    scheduler.step(val_acc)
                    lr = scheduler.optimizer.param_groups[0]['lr']

                    if val_acc > best_val:
                        best_val = val_acc
                        test_acc, test_f1, test_pr, test_re = test(test_loader)
                        model_directory = os.path.expanduser("~/.cache/mipgnn/models")
                        model_path = os.path.join(model_directory, model_name)
                        os.makedirs(model_directory, exist_ok=True)
                        torch.save(model.state_dict(), model_path)

                    log.append(
                        [epoch, train_loss, 
                            train_acc.item(), train_f1.item(), train_pr.item(), train_re.item(), 
                            val_acc.item(), val_f1.item(), val_pr.item(), val_re.item(), best_val.item(), 
                            test_acc.item(), test_f1.item(), test_pr.item(), test_re.item()])
                    print(log[-1])

                    # Break if learning rate is smaller 10**-6.
                    if lr < 0.000001 or epoch == num_epochs:
                        print([model_name, test_acc, test_f1, test_pr, test_re])
                        test_scores.append([model_name, test_acc, test_f1, test_pr, test_re])
                        log = np.array(log)
                        model_directory = os.path.expanduser("~/.cache/mipgnn/models")
                        model_log_path = os.path.join(model_directory, model_name+".log")

                        np.savetxt(model_log_path, log, delimiter=",",
                                   fmt='%1.5f')

                        fig, axes = plt.subplots(5,3, figsize=(10,10))
                        # training loss
                        axes[0,1].set_title("Train Loss")
                        axes[0,1].plot(log[:,1])

                        # training metrics
                        axes[1,0].set_title("Train Accuracy")
                        axes[1,0].plot(log[:,2])
                        axes[2,0].set_title("Train F1")
                        axes[2,0].plot(log[:,3])
                        axes[3,0].set_title("Train Precision")
                        axes[3,0].plot(log[:,4])
                        axes[4,0].set_title("Train Recall")
                        axes[4,0].plot(log[:,5])

                        # validation metrics
                        axes[1,1].set_title("Val Accuracy")
                        axes[1,1].plot(log[:,6])
                        axes[2,1].set_title("Val F1")
                        axes[2,1].plot(log[:,7])
                        axes[3,1].set_title("Val Precision")
                        axes[3,1].plot(log[:,8])
                        axes[4,1].set_title("Val Recall")
                        axes[4,1].plot(log[:,9])

                        # testing metrics
                        axes[1,2].set_title("Test Accuracy")
                        axes[1,2].plot(log[:,11])
                        axes[2,2].set_title("Test F1")
                        axes[2,2].plot(log[:,12])
                        axes[3,2].set_title("Test Precision")
                        axes[3,2].plot(log[:,13])
                        axes[4,2].set_title("Test Recall")
                        axes[4,2].plot(log[:,14])

                        plt.tight_layout()

                        plt.savefig(f"{model_name}.png")
                        break

            torch.cuda.empty_cache()
