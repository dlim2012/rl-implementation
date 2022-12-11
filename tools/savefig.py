import numpy as np
from matplotlib import pyplot as plt
import os


def savefig_mean_std(
        learning_curves,
        eval_interval,
        save_dir,
        save_name,
        plot_std=True,
        ylim=None,
        xlabel='',
        ylabel=''
    ):

    min_len = min([len(lc) for lc in learning_curves])
    for i in range(len(learning_curves)):
        learning_curves[i] = learning_curves[i][:min_len]

    learning_curve_mean = np.mean(learning_curves, axis=0)
    learning_curve_std = np.std(learning_curves, axis=0)
    #plt.figure(figsize=(8, 6), dpi=300)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(
        eval_interval * np.arange(len(learning_curve_mean)),
        learning_curve_mean,
        color='blue'
    )
    if ylim != None:
        plt.ylim(ylim)
    if plot_std:
        plt.fill_between(
            eval_interval * np.arange(len(learning_curve_mean)),
            learning_curve_mean - learning_curve_std,
            learning_curve_mean + learning_curve_std,
            alpha=0.2,
            color='blue'
        )

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, save_name))
    print(os.path.join(save_dir, save_name), 'saved')
    plt.clf()
    plt.close()