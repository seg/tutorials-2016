#!/usr/bin/env python
# -*- coding: utf 8 -*-
"""
Functions for importing Canstrat ASCII files.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patheffects as pe
from matplotlib import gridspec, spines


def norm(m):
    return m.T @ m


def misfit(d, d_pred):
    misfit = (d_pred - d).T @ (d_pred - d)
    return np.asscalar(misfit)


def plot_all(m, d, m_est, d_pred, equalize=True):
    """
    Helper function for plotting. You can ignore this.
    """
    fig = plt.figure(figsize=(10, 6))

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.plot(m)
    t = "$\mathrm{{Model}}\ \mathbf{{m}}.\ \mathrm{{norm}}\ {:.3f}$"
    ax0.set_title(t.format(norm(m)))
    ax0_mi, ax0_ma = ax0.get_ylim()

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(d, 'o', mew=0)
    ax1.set_title("$\mathrm{Data}\ \mathbf{d}$")
    ax1_mi, ax1_ma = ax1.get_ylim()

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(m, alpha=0.25)
    ax2.plot(m_est)
    t = "$\mathrm{{Estimated\ model}}\ \mathbf{{\hat{{m}}}}.\ \mathrm{{norm}}\ {:.3f}$"
    ax2.set_title(t.format(norm(m_est)))
    ax2_mi, ax2_ma = ax2.get_ylim()

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(d, 'o', mew=0, alpha=0.25)
    ax3.plot(d_pred, 'o', mew=0)
    t = "$\mathrm{{Predicted\ data}}\ \mathbf{{d}}_\mathrm{{pred}}.\ \mathrm{{misfit}}\ {:.3f}$"
    ax3.set_title(t.format(misfit(d, d_pred)))
    ax3_mi, ax3_ma = ax3.get_ylim()

    if equalize:
        ax0.set_ylim(min(ax0_mi, ax2_mi) - 0.1,
                     max(ax0_ma, ax2_ma) + 0.1)

        ax2.set_ylim(min(ax0_mi, ax2_mi) - 0.1,
                     max(ax0_ma, ax2_ma) + 0.1)

        ax1.set_ylim(min(ax1_mi, ax3_mi) - 0.1,
                     max(ax1_ma, ax3_ma) + 0.1)

        ax3.set_ylim(min(ax1_mi, ax3_mi) - 0.1,
                     max(ax1_ma, ax3_ma) + 0.1)

    plt.show()


def plot_two(m, d, m_est, d_pred, equalize=True):
    """
    Helper function for plotting. You can ignore this.
    """
    fig = plt.figure(figsize=(16, 4), facecolor='#f0f0f0')

    alpha = 0.5

    ax0 = fig.add_subplot(1, 2, 1)
    ax0.plot(m, c="#31668d", alpha=alpha)
    ax0.plot(m, 'o', c="#31668d", ms=4, alpha=alpha)
    ax0.plot(m_est, c="#462f7c", lw=1.4)
    ax0.plot(m_est, 'o', c="#462f7c", ms=4)
    ax0.set_title("Model and estimated model", size=16)
    ax0.text(35, -0.1, "m", color="#31668d", alpha=alpha+0.05, size=22)
    ax0.text(35, -0.12, "$\mathrm{{norm}} = {:.3f}$".format(norm(m)), color="#31668d", alpha=alpha+0.15, size=16)
    ax0.text(28, 0.08, "m_est", color="#462f7c", size=22)
    ax0.text(28, 0.05, "$\mathrm{{norm}} = {:.3f}$".format(norm(m_est)), color="#462f7c", size=16)

    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(d, 'o', c="#31668d", mew=0, ms=5, alpha=alpha)
    ax1.plot(d, c="#31668d", alpha=alpha)
    ax1.plot(d_pred, 'o', c="#462f7c", ms=5, mew=0)
    ax1.plot(d_pred, c="#462f7c", lw=1.4)
    ax1.set_title("Data and predicted data", size=16)
    ax1.text(11.3, 0.13, "d", color="#31668d", alpha=alpha+0.05, size=22)
    ax1.text(14, 0.13, "d_pred", color="#462f7c", size=22)
    ax1.text(14, 0.096, "$\mathrm{{misfit}} = {:.3f}$".format(misfit(d, d_pred)), color="#462f7c", size=16)

    for ax in fig.axes:
        ax.xaxis.label.set_color('#777777')
        ax.tick_params(axis='y', colors='#777777')
        ax.tick_params(axis='x', colors='#777777')
        for child in ax.get_children():
            if isinstance(child, spines.Spine):
                child.set_color('#aaaaaa')

    plt.savefig('figures/figure2.png', dpi=200, facecolor=fig.get_facecolor())
    plt.show()


def add_subplot_axes(ax, rect, axisbg='w'):
    """
    Facilitates the addition of a small subplot within another plot.

    From: http://stackoverflow.com/questions/17458580/
    embedding-small-plots-inside-subplots-in-matplotlib

    License: CC-BY-SA

    Args:
        ax (axis): A matplotlib axis.
        rect (list): A rect specifying [left pos, bot pos, width, height]
    Returns:
        axis: The sub-axis in the specified position.
    """
    def axis_to_fig(axis):
        fig = axis.figure

        def transform(coord):
            a = axis.transAxes.transform(coord)
            return fig.transFigure.inverted().transform(a)

        return transform

    fig = plt.gcf()
    left, bottom, width, height = rect
    trans = axis_to_fig(ax)
    x1, y1 = trans((left, bottom))
    x2, y2 = trans((left + width, bottom + height))
    subax = fig.add_axes([x1, y1, x2 - x1, y2 - y1])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)

    return subax


def plot_gmd(G, m, d):
    fig = plt.figure(figsize=(12, 6), facecolor='#f0f0f0')

    gs = gridspec.GridSpec(5, 8)

    # Set up axes.
    axw = plt.subplot(gs[0, :5])    # Wavelet.
    axg = plt.subplot(gs[1:4, :5])  # G
    axv = plt.subplot(gs[4, :5])    # Other wavelet.
    axm = plt.subplot(gs[:, 5])     # m
    axe = plt.subplot(gs[:, 6])     # =
    axd = plt.subplot(gs[1:4, 7])   # d

    cax = add_subplot_axes(axg, [-0.14, 0.22, 0.03, 0.5])

    params = {'ha': 'center',
              'va': 'bottom',
              'size': 40,
              'weight': 'bold',
              }

    axw.plot(G[5], 'o', c='r', mew=0)
    axw.plot(G[5], 'r', alpha=0.4)
    axw.locator_params(axis='y', nbins=3)
    axw.text(1, 0.8, "one row of G", color='#333333')
    axw.set_ylim(-0.7, 1.3)

    cyan = '#26c3a4'

    im = axg.imshow(G, cmap='viridis',
                    aspect='auto',
                    interpolation='none')
    axg.text(45, G.shape[0]//2, "G", color='w', **params)
    axg.axhline(5, color='r')
    axg.axhline(20, color=cyan)
    cb = plt.colorbar(im, cax=cax)

    axv.plot(G[20], 'o', c=cyan, mew=0)
    axv.plot(G[20], cyan, alpha=0.4)
    axv.locator_params(axis='y', nbins=3)
    axv.text(1, 0.8, "another row of G", color='#333333')
    axv.set_ylim(-0.7, 1.3)

    y = np.arange(m.size)
    axm.plot(m, y, 'o', c='r', mew=0)
    axm.plot(m, y, c='r', alpha=0.4)
    # axm.imshow(np.expand_dims(m, 1), cmap='viridis', interpolation='none', aspect='auto')
    txt = axm.text(0, m.size//2, "m", color='#333333', **params)
    txt.set_path_effects([pe.withStroke(linewidth=5, foreground='w')])
    axm.invert_yaxis()
    axm.set_xlim(-0.2, 0.2)
    axm.locator_params(axis='x', nbins=3)

    axe.set_frame_on(False)
    axe.set_xticks([])
    axe.set_yticks([])
    axe.text(0.5, 0.5, "=", color='#333333', **params)
    axe.text(0.5, 0.8, "forward\nproblem", color='#31668d', size=16, ha='center')
    axe.arrow(0.2, 0.75, 0.6, 0, head_width=0.03, head_length=0.2, fc='#31668d', ec='#31668d')
    axe.text(0.5, 0.2, "inverse\nproblem", color='#31668d', size=16, ha='center', va='top')
    axe.arrow(0.8, 0.25, -0.6, 0, head_width=0.03, head_length=0.2, fc='#31668d', ec='#31668d')

    y = np.arange(d.size)
    axd.plot(d, y, 'o', c='#462f7c', mew=0)
    axd.plot(d, y, c='#462f7c', alpha=0.4)
    axd.plot(d[5], y[5], 'o', c='r', mew=0, ms=10)
    # axd.imshow(np.expand_dims(d, 1), cmap='viridis', interpolation='none', aspect='auto')
    txt = axd.text(0, d.size//2, "d", color='#333333', **params)
    txt.set_path_effects([pe.withStroke(linewidth=5, foreground='w')])
    axd.invert_yaxis()
    axd.set_xlim(-0.2, 0.2)
    axd.locator_params(axis='x', nbins=3)

    for ax in fig.axes:
        ax.xaxis.label.set_color('#777777')
        ax.tick_params(axis='y', colors='#777777')
        ax.tick_params(axis='x', colors='#777777')
        for child in ax.get_children():
            if isinstance(child, spines.Spine):
                child.set_color('#aaaaaa')

    cb.outline.set_edgecolor('white')
    cax.xaxis.label.set_color('#ffffff')
    cax.tick_params(axis='y', colors='#ffffff')
    cax.tick_params(axis='x', colors='#ffffff')

    fig.tight_layout()
    plt.savefig("figures/figure1.png", dpi=200, facecolor=fig.get_facecolor())
    plt.show()

    return
