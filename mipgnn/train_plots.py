import sys
import matplotlib.pyplot as plt
import numpy as np

def metric_plot(x, index_col, cols, labels, title, xlabel="epoch", ylabel="metric"):
    f,ax = plt.subplots(1, 1)
    for i,c in enumerate(cols):
        ax.plot(x[:,index_col], x[:,c], label=labels[i])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    return f,ax

def training_loss_plot(x):
    return metric_plot(x, 0, [1,6], 
                       title="Training and validation loss",
                       labels=["train_loss","val_loss"],
                       ylabel="loss")

def accuracy_plot(x):
    return metric_plot(x, 0, [2,7], 
                       title="Training and validation accuracy",
                       labels=["train_acc","val_acc"],
                       ylabel="accuracy")


def f1_plot(x):
    return metric_plot(x, 0, [3,8], 
                       title="Training and validation accuracy",
                       labels=["train_f1","val_f1"],
                       ylabel="accuracy")
def precision_plot(x):
    return metric_plot(x, 0, [4,9], 
                       title="Training and validation precision",
                       labels=["train_pr","val_pr"],
                       ylabel="precision")

def recall_plot(x):
    return metric_plot(x, 0, [5,10], 
                       title="Training and validation recall",
                       labels=["train_re","val_re"],
                       ylabel="recall")


if __name__=="__main__":
    history_log_path = sys.argv[1]

    x = np.genfromtxt(history_log_path, delimiter=",")
    print(x.shape)

    f,ax = training_loss_plot(x)
    f,ax = accuracy_plot(x)
    f,ax = f1_plot(x)
    f,ax = precision_plot(x)
    f,ax = recall_plot(x)
    plt.show()
