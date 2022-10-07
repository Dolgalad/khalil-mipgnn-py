import sys
import argparse
import json
import time
from  datetime import datetime
date_fmt = "%d/%m/%YT%H:%M:%S.%f"

from progress.bar import Bar

import matplotlib.pyplot as plt

import os
import os.path as osp

from sklearn.model_selection import train_test_split

from torchmetrics import F1Score, Precision, Recall, Accuracy

#from torch_geometric.data import (InMemoryDataset, Data)
from torch_geometric.loader import DataLoader

import numpy as np
#import pandas as pd
import torch
import torch.nn.functional as F

from mipgnn.gnn_models import manager as gnn_manager

from mipgnn.predict import create_data_object

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from mipgnn.datasets import GraphDataset, MyData, MyTransform

def is_int(s):
    try:
        i = int(s)
        return True
    except:
        return False

def train_epoch_model(model, dataloader, optimizer, device=None, batch_size=10):
    #if device is None:
    #    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    model.train()

    # loss_all = 0
    zero = torch.tensor([0]).to(device)
    one = torch.tensor([1]).to(device)

    loss_all = 0

    for data in dataloader:
        data = data.to(device)
        #print("data shape, ", data.shape)


        y = data.y_real
        y = torch.where(y <= bias_threshold, zero, one).to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, y)
        loss.backward()
        loss_all += batch_size * loss.item()
        optimizer.step()

    return loss_all / len(train_dataset)

def train_model(model, epochs, train_loader, val_loader, optimizer, device=None, batch_size=10, metrics=[], model_name="generic_model"):
    best_val = 0.0
    test_acc = 0.0
    test_f1 = 0.0
    test_re = 0.0
    test_pr = 0.0

    # training history
    history = []
    
    # target directory
    model_directory = os.path.expanduser("~/.cache/mipgnn/models")
    model_log_path = os.path.join(model_directory, model_name+".log")
    model_path = os.path.join(model_directory, model_name)
    os.makedirs(model_directory, exist_ok=True)
 
    with Bar(f"Training {model_name}: ", max=epochs) as bar:
        for epoch in range(epochs):
            train_loss = train_epoch_model(model, train_loader, optimizer, batch_size=batch_size)
    
            [train_loss, train_acc,train_f1,train_pr,train_re] = evaluate_model(model, train_loader, metrics=metrics)
    
            [val_loss,val_acc,val_f1,val_pr,val_re] = evaluate_model(model, val_loader, metrics=metrics)
            scheduler.step(val_acc)
            lr = scheduler.optimizer.param_groups[0]['lr']
    
            if val_acc > best_val:
                best_val = val_acc
                #[test_acc,test_f1,test_pr,test_re] = evaluate_model(model, test_loader, metrics=metrics)
                torch.save(model.state_dict(), model_path)
    
            history.append(
                [epoch, train_loss, 
                    train_acc.item(), train_f1.item(), train_pr.item(), train_re.item(), 
                    val_loss,
                    val_acc.item(), val_f1.item(), val_pr.item(), val_re.item(), best_val.item(), 
                    ])
                    #test_acc.item(), test_f1.item(), test_pr.item(), test_re.item()])
    
            # Break if learning rate is smaller 10**-6.
            if lr < 0.000001 or epoch == num_epochs:
                #print([model_name, test_acc, test_f1, test_pr, test_re])
                #test_scores.append([model_name, test_acc, test_f1, test_pr, test_re])
                #history = np.array(history)
                #np.savetxt(model_log_path, history, delimiter=",",
                #           fmt='%1.5f')
                break

            bar.next()
    history = np.array(history)
    print(f"Saving training history to {model_log_path}")
    np.savetxt(model_log_path, history, delimiter=",",
                           fmt='%1.5f',
                           header="epoch,train_loss,train_acc,train_f1,train_pr,train_re,val_loss,val_acc,val_f1,val_pr,val_re,best_val_acc")

    return np.array(history)



@torch.no_grad()
def evaluate_model(model, dataloader, device=None, metrics=[]):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    zero = torch.tensor([0]).to(device)
    one = torch.tensor([1]).to(device)

    metrics = [metric.to(device) for metric in metrics]

    first = True
    loss_all  = 0.
    for data in dataloader:
        data = data.to(device)
        pred = model(data)
        
        
        y = data.y_real

        y = torch.where(y <= bias_threshold, zero, one)#.to(device)

        loss = F.nll_loss(pred, y)

        pred = pred.max(dim=1)[1]

        #print("in evaluate_model", len(data), len(dataloader), pred.shape, y.shape)
        loss_all += len(data) * loss.item()



        if not first:
            pred_all = torch.cat([pred_all, pred])
            y_all = torch.cat([y_all, y])
        else:
            pred_all = pred
            y_all = y
            first = False

    return [loss_all/len(dataloader)]+[m(pred_all, y_all) for m in metrics]

    return acc(pred_all, y_all), f1(pred_all, y_all), pr(pred_all, y_all), re(pred_all, y_all)




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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model that you want to train")
    parser.add_argument("--model-args", type=str, help="model constructor arguments as JSON", default=gnn_manager.DEFAULT_MODEL_ARGS_STR)
    parser.add_argument("--name", type=str, help="name of the trained model", default="")
    parser.add_argument("--train-data", type=str, help="path to training data")
    parser.add_argument("--validation-split", type=float, help="validation split (default=0.)", default=0.)
    parser.add_argument("--bias-threshold", type=float, help="bias threshold (default=0.)", default=0.)
    parser.add_argument("--batch-size", type=int, help="batch size (default=10)", default=10)
    parser.add_argument("--epochs", type=int, help="number of epochs (default=30)", default=30)
    parser.add_argument("--learning-rate", type=float, help="learning rate (default=0.001)", default=0.001)
    parser.add_argument("--min-learning-rate", type=float, help="minimum learning rate (default=0.0000001)", default=0.0000001)
    parser.add_argument("--patience", type=int, help="scheduler patience (default=10)", default=10)
    parser.add_argument("--train-test-split", type=float, help="split into training and testing sets (default=0.2)", default=0.2)
    parser.add_argument("--list-models", action="store_true", help="show a list of model classes")


    args = parser.parse_args()

    # starting datetime
    start_dt = datetime.now()

    # metrics 
    metrics = [Accuracy(num_classes=2),
               F1Score(num_classes=2, average="macro"),
               Precision(num_classes=2, average="macro"),
               Recall(num_classes=2, average="macro"),
            ]

    # Prepare data.
    bias_threshold = args.bias_threshold
    batch_size = 1#args.batch_size
    num_epochs = args.epochs

    # processed dataset directory
    processed_data_root = os.path.expanduser("~/.cache/mipgnn/datasets")

    # training data
    train_data_path = args.train_data
    if not os.path.exists(train_data_path):
        print(f"Train data path not found : {train_data_path}")

    name_train = os.path.basename(train_data_path)
    print(f"Loading training data {name_train} from {train_data_path}")
    train_dataset = GraphDataset(name_train, processed_data_root, train_data_path, bias_threshold,
                                 transform=MyTransform()).shuffle()
 

    # model
    model_args = json.loads(args.model_args)
    print(f"Training model class: {args.model}")
    print(f"Model arguments:")
    print(json.dumps(model_args, indent=4, sort_keys=True))
    
    if is_int(args.model):
        model_cls = gnn_manager.get_model_class_instance(index=int(args.model))
    else:
        model_cls = gnn_manager.get_model_class_instance(name=args.model)
    model = model_cls(**model_args).to(device)
    model_name = args.name
    if len(model_name)==0:
        model_name = f"{model_cls._class_prefix}_{name_train}"

    print(f"Model name: {model_name}")
    
    # training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.8, patience=args.patience,
                                                           min_lr=args.min_learning_rate)
 

    train_index, val_index = train_test_split(list(range(0, len(train_dataset))), test_size=args.train_test_split)
    #train_index, val_index = train_test_split(train_index, test_size=.2)

    val_dataset = train_dataset[val_index].shuffle()
    #test_dataset = train_dataset[test_index].shuffle() #test_dataset.shuffle()
    train_dataset = train_dataset[train_index].shuffle()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # train the model
    torch.cuda.empty_cache()
    
    training_start_t = time.time()
    history = train_model(model, args.epochs, train_loader, val_loader, optimizer, model_name=model_name, metrics=metrics)
    training_end_t = time.time()
    t_train = training_end_t - training_start_t

    # save the models training arguments
    model_data = {"name": model_name,
            "datetime": datetime.strftime(start_dt, date_fmt),
            "model_class":model_cls.__name__,
            "model_args": json.loads(args.model_args),
            "epochs": args.epochs,
            "learining_rate": args.learning_rate,
            "patience": args.patience,
            "min_learning_rate": args.min_learning_rate,
            "train_dataset": name_train,
            "train_test_split": args.train_test_split,
            "times":{
                "training_start": training_start_t,
                "training_end": training_end_t,
                "training_seconds": t_train,
            }
            }
    model_data_path= os.path.expanduser(f"~/.cache/mipgnn/models/{model_name}.json")
    with open(model_data_path, "w") as f:
        f.write(json.dumps(model_data, indent=4, sort_keys=True))

    torch.cuda.empty_cache()
   


