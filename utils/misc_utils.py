"""
  misc utils for plotting and any application specific metrics
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib import cm

def compute_grad_norm(p_list):
    grad_norm = 0
    for p in p_list:
        param_g_norm = p.grad.detach().data.norm(2)
        grad_norm += param_g_norm.item()**2
    grad_norm = grad_norm**0.5
    return grad_norm

def show(u, ax, fig, rescale=None):
    ''' plot some output function to keep track of during logging '''
    h = ax.imshow(u.T, interpolation='nearest', cmap='rainbow',
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    ax.tick_params(labelsize=15)

def vis_fields(fields, params):
    targets, pred = fields
    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(1,2,1)
    show(targets, ax1, fig)
    ax1.set_title("target")

    ax2 = fig.add_subplot(1,2,2)
    show(pred, ax2, fig)
    ax2.set_title("pred")

    fig.tight_layout()

    return fig
