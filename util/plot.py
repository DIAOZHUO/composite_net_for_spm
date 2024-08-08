import os
import matplotlib.pyplot as plt
import SPMUtil as spmu
import numpy as np
import seaborn as sn


def plot_matrix(confusion_matrix, names, normalize=True):
    print("iou_thres:", confusion_matrix.iou_thres)
    print("conf:", confusion_matrix.conf)
    array = confusion_matrix.matrix / ((confusion_matrix.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc, nn = confusion_matrix.nc, len(names)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
    labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
    ticklabels = (list(names) + ['background']) if labels else 'auto'
    sn.heatmap(array,
                   ax=ax,
                   annot=nc < 30,
                   annot_kws={
                       'size': 8},
                   cmap='Blues',
                   fmt='.2f' if normalize else '.0f',
                   square=True,
                   vmin=0.0,
                   xticklabels=ticklabels,
                   yticklabels=ticklabels).set_facecolor((1, 1, 1))
    title = 'Confusion Matrix' + ' Normalized' * normalize
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    plt.show()