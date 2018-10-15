#!/usr/bin/env python

"""
Simple script to replot the performance of different RBM variants based on saved out data.
"""

import numpy as np
import matplotlib.pyplot as plt

# styling
import matplotlib as mpl
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


if __name__ == '__main__':

    experiments = [
        ['#1f77b4',    'Canonical RBM',                                  'f3_canonical_rbm'],
        ['#9467bd',    'Directed RBM',                                   'f3_directed_rbm'],
        ['#d62728',   r'Directed RBM with constant w$_{F}$',             'f3_directed_rbm_with_constant_w_F'],
        ['#ff7f0e',   r'Directed RBM with constant w$_{B}$',             'f3_directed_rbm_with_constant_w_B'],
        ['lightgray', r'Directed RBM with constant w$_{F}$ and w$_{B}$', 'f3_directed_rbm_with_constant_w_F_and_w_B']
    ]

    ddir = '../data/results_'

    # initialise figure
    fig, ax = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)

    for ii, (color, label, fname) in enumerate(experiments):

        # --------------------------------------------------------------------------------
        # load data

        fpath = ddir + fname + '.npz'
        data = np.load(fpath)
        samples = data['samples']
        loss = data['loss']

        # --------------------------------------------------------------------------------
        # plot loss versus time

        if samples[0] == 0:
            samples[0] = 1 # show first data point if x-axis is log-scaled

        ax.errorbar(x=samples, y=loss.mean(axis=0),
                    yerr=loss.std(axis=0), errorevery=1,
                    color=color, ecolor=color, alpha=0.9,
                    linewidth=2, linestyle='-',
                    label=label)

        # annotate final loss
        final_sample = samples[-1]
        final_loss = loss[:,-1]
        ax.annotate(s='{:.3f}'.format(final_loss.mean()),
                    xy=(final_sample, final_loss.mean()), xycoords='data',
                    xytext=(5, 0), textcoords='offset points',
                    fontsize='xx-small')

        # layout plot
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_ylim(0, 0.55)
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Mean squared error')
        ax.legend(loc=1, fontsize='xx-small')

        fig.tight_layout()
        fig.savefig('../figures/performance_vs_time.pdf', dpi=300)
        fig.savefig('../figures/performance_vs_time.svg', dpi=300)

        # --------------------------------------------------------------------------------
        # plot comparison of final loss

        ax2.bar(ii, final_loss.mean(), yerr=final_loss.std(), color=color, alpha=0.9)

        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels([label for (color, label, fname) in experiments], rotation=90)
        ax2.set_ylabel('Mean squared error')

        fig2.tight_layout()
        fig2.savefig('../figures/final_performance.pdf', dpi=300)
        fig2.savefig('../figures/final_performance.svg', dpi=300)
