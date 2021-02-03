#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for Data Visualization

"""
import seaborn as sns
from matplotlib import pyplot as plt

def plot_metricsNN(all_results):
    fig, axs = plt.subplots(figsize=(15,3), ncols=3)
    sns.lineplot(x='Epochs', y='Accuracy', hue='Batch', data=all_results, ci=None, ax=axs[0])
    sns.lineplot(x='Epochs', y='Precision', hue='Batch', data=all_results, ci=None, ax=axs[1])
    sns.lineplot(x='Epochs', y='Recall', hue='Batch', data=all_results, ci=None, ax=axs[2])
    axs[0].set_title('Accuracy')
    axs[1].set_title('Precision')
    axs[2].set_title('Recall')
    return fig