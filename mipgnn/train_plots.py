import sys
import matplotlib.pyplot as plt
import numpy as np


def training_loss_plot(x):
    f,ax = plt.subplots(1, 1)
    ax.plot(x[:,0], x[:,1], label="train_loss")
    ax.plot(x[:,0], x[:,6], label="val_loss")

    return f,ax

if __name__=="__main__":
    history_log_path = sys.argv[1]

    x = np.genfromtxt(history_log_path, delimiter=",")
    print(x.shape)

    f,ax = training_loss_plot(x)

    plt.show()
