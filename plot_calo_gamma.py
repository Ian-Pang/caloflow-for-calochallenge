""" Helper script to plot generated calo data

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

    Layout inspired by
    "CaloGAN: Simulating 3D High Energy Particle Showers in Multi-LayerElectromagnetic
     Calorimeters with Generative Adversarial Networks"
    by Michela Paganini, Luke de Oliveira, and Benjamin Nachman
    arxiv:1712.10321
    https://github.com/hep-lbdl/CaloGAN
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm

# for nearest neighbor plots:
from data import CaloDataset
from sklearn.neighbors import NearestNeighbors

irange = range
from torchvision.utils import make_grid

# hard coded size of calo layers (= number of cells):
SIZES = [1, 8, 10, 16, 10, 19, 1, 5, 1, 5]

# threshold in MeV below which voxels are assumed to be 0.
# LOWER_THRESHOLD = 0. # moved to function argument

def layer_std(layer1, layer2, total):
    """ helper function for standard deviation of layer depth"""
    term1 = (layer1 + 4.*layer2) / total
    term2 = ((layer1 + 2.*layer2)/total)**2
    return np.sqrt(term1-term2)

def plot_calo_batch(tensor, fp, layer, ncol=8, vmin=1e-2, vmax=None, save_it=True,
                    lower_threshold=0.):
    """ combination of torchvision.utils.save_image and plot_image()
        of CaloGAN github

        tensor (torch.tensor): input data of shape (nbatch, 3, 96) or (nbatch, 12, 12),
                               or (nbatch, 12, 6)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        ncol (int): how many columns to plot
        vmin (float): minimal value in colorbar
        vmax (float): maximal value in colorbar
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        lower_threshold (float): threshold to be applied to data before plotting
    """

    x_len = SIZES[2*layer + 1]
    y_len = SIZES[2*layer]
    # number of pixels to be drawn to have right aspect ratio
    scale_factor = 1
    plot_sizes = scale_factor * np.array([8, 1, 16, 10, 19, 10, 5 ,1, 5, 1])
    x_dim = plot_sizes[2*layer + 1]
    y_dim = plot_sizes[2*layer]
    #print('x_dim: ',x_dim)
    #print('y_dim: ',y_dim)
    padding = 5
    #print('Tensor size: ',list(tensor.size()))
    ndarr = tensor.clamp_(0., 1e5).to('cpu').numpy()
    ndarr = np.where(ndarr < lower_threshold, np.zeros_like(ndarr), ndarr)
    assert len(ndarr.shape) == 3

    n_sam = ndarr.shape[0]
    #print('ndarr: ', ndarr)
    #print('ndarr shape: ', ndarr.shape)
    x_sam = min(ncol, n_sam)
    #y_sam = int((np.ceil(float(n_sam) / x_sam))/100)
    y_sam = int(np.ceil(float(n_sam) / x_sam))
    #print('x_sam: ',x_sam)
    #print('y_sam: ',y_sam)
    grid = np.zeros((y_sam*y_dim*y_len + (y_sam-1)*padding,
                     x_sam*x_dim*x_len + (x_sam-1)*padding))
    k = 0
    for y in range(y_sam):
        for x in range(x_sam):
            if k >= n_sam:
                break
            x_start = x*(padding + x_dim*x_len)
            x_end = (x+1)*x_dim*x_len + x*padding
            y_start = y*(padding + y_dim*y_len)
            y_end = (y+1)*y_dim*y_len + y*padding
            local_image = ndarr[k].copy()
            #print('start and end: ',x_start,x_end,y_start,y_end)
            # enlarge to have right aspect ratio
            local_image_large = np.repeat(local_image, y_dim, axis=0)
            local_image_large = np.repeat(local_image_large, x_dim, axis=1)

            grid[y_start:y_end, x_start:x_end] = local_image_large
            k += 1
    plt.imshow(grid, interpolation='nearest', aspect=1.,
               norm=LogNorm(vmin, vmax))
    #print('grid shape: ',grid.shape)
    for y in range(1, y_sam):
        plt.plot([0., grid.shape[1]],
                 [y*(y_dim*y_len+padding) + (padding >> 1) - padding,
                  y*(y_dim*y_len+padding) + (padding >> 1) - padding],
                 color='k', lw=1.)
    for x in range(1, x_sam):
        plt.plot([x*(x_dim*x_len+padding) + (padding >> 1) - padding,
                  x*(x_dim*x_len+padding) + (padding >> 1) - padding],
                 [0., grid.shape[0]], color='k', lw=1.)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.xlim([0., grid.shape[1]])
    plt.ylim([0., grid.shape[0]])
    cbar = plt.colorbar(fraction=0.0455)
    cbar.set_label(r'Energy (MeV)', y=0.83)
    cbar.ax.tick_params()
    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_average_voxel(tensor, fp, layer, vmin=1e-2, vmax=None, save_it=True,
                       lower_threshold=0.):
    """ Plots the average value per voxel of given batch, cf CaloGAN [1712.10321] Figs. 6, 7, 8
        tensor (torch.tensor): input data of shape (nbatch, 3, 96) or (nbatch, 12, 12),
                               or (nbatch, 12, 6)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        vmin (float): minimal value in colorbar
        vmax (float): maximal value in colorbar
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        lower_threshold (float): threshold to be applied to data before plotting
    """
    x_len = SIZES[2*layer + 1]
    y_len = SIZES[2*layer]
    # number of pixels to be drawn to have right aspect ratio
    scale_factor = 1
    plot_sizes = scale_factor * np.array([32, 1, 4, 4, 2, 4])
    x_dim = plot_sizes[2*layer + 1]
    y_dim = plot_sizes[2*layer]

    ndarr = tensor.clamp_(0., 1e5).to('cpu').numpy()
    ndarr = np.where(ndarr < lower_threshold, np.zeros_like(ndarr), ndarr)
    assert len(ndarr.shape) == 3

    av_vox = ndarr.mean(axis=0)
    # enlarge to have right aspect ratio
    av_vox_large = np.repeat(av_vox, y_dim, axis=0)
    av_vox_large = np.repeat(av_vox_large, x_dim, axis=1)

    plt.figure()
    plt.imshow(av_vox_large, interpolation='nearest', aspect=1.,
               norm=LogNorm(vmin, vmax))
    #frame = plt.gca()

    xticks = range(0, x_dim*SIZES[layer*2 + 1], x_dim)+ (x_dim-1)/2
    xlabels = range(SIZES[layer*2 + 1])
    yticks = range(0, y_dim*SIZES[layer*2], y_dim) + (y_dim-1)/2
    ylabels = range(SIZES[layer*2])
    if layer == 0:
        xticks = xticks[::10]
        xlabels = xlabels[::10]
        #plt.text(-30, 65, 'GEANT4', rotation=90, fontsize=24)
        plt.text(-30, 65, 'CaloFlow', rotation=90, fontsize=24)
        #plt.text(-30, 50, 'CaloFlow teacher', rotation=90, fontsize=24, va='center')
        #plt.text(-30, 50, 'CaloFlow student', rotation=90, fontsize=24, va='center')
        #plt.text(-30, 65, 'CaloGAN', rotation=90, fontsize=24)
    plt.xticks(xticks, xlabels, fontsize=16)
    plt.yticks(yticks, ylabels, fontsize=16)
    plt.xlabel(r'$\alpha$ Cell ID', fontsize=16)
    plt.ylabel(r'$r$ Cell ID', fontsize=16)

    #frame.axes.get_xaxis().set_visible(False)
    #frame.axes.get_yaxis().set_visible(False)
    cbar = plt.colorbar(fraction=0.0455)
    cbar.set_label(r'Energy (MeV)', y=0.83, fontsize=16)
    cbar.ax.tick_params()
    plt.title('Layer '+str(layer), fontsize=18)
    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_layer_energy(tensor, fp, layer, save_it=True, plot_ref=None, epoch_nr=None,
                      lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the energy deposit in the full layer for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 3, 96) or (nbatch, 12, 12),
                               or (nbatch, 12, 6)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    energies = data.sum(axis=(1, 2))
    if layer == 0:
        bins = np.logspace(-2, 2, 100)
        x_max = 40
        axis_label = r'$E_0$ (GeV)'
    elif layer == 1:
        bins = np.logspace(-1, 3, 100)
        x_max = 140
        axis_label = r'$E_1$ (GeV)'
    elif layer == 2:
        bins = np.logspace(-2, 2, 100)
        x_max = 100
        axis_label = r'$E_2$ (GeV)'
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if ref_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
            raise ValueError(
                "lower_threshold {} not in pre-computed set for plot_ref".format(ref_thres))
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            #reference_data = pd.read_hdf('reference_energy_thresholds.hdf')
            reference = reference_data.loc[plot_ref]\
                ['energy_layer_{}_{:1.0e}'.format(str(layer), ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
                leg_loc = 'upper left'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
                leg_loc = 'lower left'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
                leg_loc = 'upper right'
            _ = plt.hist(reference / 1000., bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_GAN]\
                ['energy_layer_{}'.format(str(layer))]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
                leg_loc = 'upper left'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
                leg_loc = 'lower left'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
                leg_loc = 'upper right'
            _ = plt.hist(reference / 1000., bins=bins, histtype='step',
                         linewidth=2., alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(energies / 1000, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(10., 0.1, 'epoch '+str(epoch_nr))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(xmax=x_max)
    plt.ylim([1e-6, 2e1])
    #plt.legend(loc=leg_loc if plot_ref is not None else 'upper right', fontsize=20)
    plt.legend(fontsize=20)

    plt.xlabel(axis_label)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_total_energy(tensor, fp, save_it=True, plot_ref=None, epoch_nr=None,
                      lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the total energy deposit in all 3 layers for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    energies = data.sum(axis=-1)
    bins = np.linspace(0, 120, 50)
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if ref_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
            raise ValueError(
                "lower_threshold {} not in pre-computed set for plot_ref".format(ref_thres))
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            #reference_data = pd.read_hdf('reference_energy_thresholds.hdf')
            reference = reference_data.loc[plot_ref]\
                ['energy_layer_0_{:1.0e}'.format(ref_thres)]

            reference += reference_data.loc[plot_ref]\
                ['energy_layer_1_{:1.0e}'.format(ref_thres)]
            reference += reference_data.loc[plot_ref]\
                ['energy_layer_2_{:1.0e}'.format(ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference / 1000., bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['energy_layer_0']

            reference += reference_data.loc[plot_ref]['energy_layer_1']
            reference += reference_data.loc[plot_ref]['energy_layer_2']
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference / 1000., bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(energies / 1000, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(110., 0.02, 'epoch '+str(epoch_nr))
    plt.yscale('log')
    plt.ylim([5e-6, 2e-1])
    #plt.legend(loc='lower left', fontsize=20)
    plt.legend(fontsize=20)

    plt.xlabel(r'$\hat{E}_\mathrm{tot}$ (GeV)')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_energy_fraction(tensor, fp, layer, save_it=True, plot_ref=None, epoch_nr=None,
                         use_log=False, lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the energy fraction deposited in the given layer for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        use_log (bool): if True, axes are in log scale
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    total_energies = data.sum(axis=-1)

    if layer == 0:
        energies = data[..., :288].sum(axis=-1)
        if use_log:
            bins = np.logspace(-4, 0, 100)
            text_loc = [0.1, 50.]
        else:
            bins = np.linspace(0., 0.4, 100)
            text_loc = [0.3, 5.]
        axis_label = r'$E_0 / \hat{E}_\mathrm{tot}$'
    elif layer == 1:
        energies = data[..., 288:432].sum(axis=-1)
        if use_log:
            bins = np.logspace(-1, 0, 100)
            text_loc = [0.1, 1.]
        else:
            bins = np.linspace(0., 1., 100)
            text_loc = [0., 4.]
        axis_label = r'$E_1 / \hat{E}_\mathrm{tot}$'
    elif layer == 2:
        energies = data[..., 432:].sum(axis=-1)
        if use_log:
            bins = np.logspace(-4, 1, 100)
            text_loc = [0.5, 10.]
        else:
            bins = np.linspace(0., 0.008, 100)
            text_loc = [0.005, 1500.]
        axis_label = r'$E_2 / \hat{E}_\mathrm{tot}$'
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if ref_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
            raise ValueError(
                "lower_threshold {} not in pre-computed set for plot_ref".format(ref_thres))
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference = reference_data.loc[plot_ref]\
                ['energy_layer_{}_{:1.0e}'.format(str(layer), ref_thres)]

            reference_total = reference_data.loc[plot_ref]\
                ['energy_layer_0_{:1.0e}'.format(ref_thres)]
            reference_total += reference_data.loc[plot_ref]\
                ['energy_layer_1_{:1.0e}'.format(ref_thres)]
            reference_total += reference_data.loc[plot_ref]\
                ['energy_layer_2_{:1.0e}'.format(ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference / reference_total, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['energy_layer_{}'.format(str(layer))]

            reference_total = reference_data.loc[plot_ref]['energy_layer_0']
            reference_total += reference_data.loc[plot_ref]['energy_layer_1']
            reference_total += reference_data.loc[plot_ref]['energy_layer_2']
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference / reference_total, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(energies / total_energies, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(*text_loc, 'epoch '+str(epoch_nr))
    if use_log:
        plt.yscale('log')
        plt.xscale('log')
    #plt.legend(loc='upper right', fontsize=20)
    plt.legend(fontsize=20)

    plt.xlabel(axis_label)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_layer_sparsity(tensor, fp, layer, save_it=True, plot_ref=None, epoch_nr=None,
                        threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the sparsity (number of non-zero voxel) per layer
        tensor (torch.tensor): input data of shape (nbatch, 3, 96) or (nbatch, 12, 12),
                               or (nbatch, 12, 6)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        threshold (float): threshold to be applied to data before determining sparsity
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    sparsity = [(x > threshold).mean((0, 1)) for x in data]
    bins = np.linspace(0, 1, 20)
    if layer == 1:
        loc = 'upper center'
    else:
        loc = 'upper right'
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = threshold
        if ref_thres not in [0., 1e-1, 1e-2, 1e-3, 1e-4]:
            raise ValueError(
                "threshold {} not in pre-computed set for plot_ref".format(ref_thres))
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference = reference_data.loc[plot_ref]\
                ['sparsity_layer_{}_{:1.0e}'.format(str(layer), ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['sparsity_layer_{}'.format(str(layer))]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(sparsity, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(0.5, 2. if layer == 2 else 4., 'epoch '+str(epoch_nr))
    plt.legend(loc=loc, fontsize=20)

    plt.xlabel('Sparsity in Layer {}'.format(layer))

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_energy_distribution_layer(tensor, energyarray, fp, save_it=True, epoch_nr=None,
                                   use_log=False, lower_threshold=0.):
    """ plots the energy deposited in each layer for a given batch,
        normalized to Etot (total Energy)
        tensor (torch.tensor): input data of shape (nbatch, 504)
        energyarray (torch.tensor): batch of input Etot (nbatch,)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        epoch_nr (int): if not None, epoch nr will be shown in plot
        use_log (bool): if True, axes are in log scale
        lower_threshold (float): threshold to be applied to data before plotting
    """
    energyarray = energyarray.to('cpu').numpy().squeeze(axis=-1)
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    energies_0 = data[..., :288].sum(axis=-1)
    energies_1 = data[..., 288:432].sum(axis=-1)
    energies_2 = data[..., 432:].sum(axis=-1)

    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    plt.scatter(energyarray, energies_0 / (1e3*energyarray), s=0.5, label='layer 0')
    plt.scatter(energyarray, energies_1 / (1e3*energyarray), s=0.5, label='layer 1')
    plt.scatter(energyarray, energies_2 / (1e3*energyarray), s=0.5, label='layer 2')
    plt.plot([0., 100.], [1., 1.], linestyle='dashed', color='k', linewidth=1.)

    if epoch_nr is not None:
        plt.text(90., 0.9, 'epoch '+str(epoch_nr))
    if use_log:
        plt.yscale('log')

    plt.legend(loc='upper right', markerscale=10., scatterpoints=1, fontsize=20)

    plt.xlabel(r'$E_\mathrm{tot}$')
    plt.ylabel(r'$E_i /E_\mathrm{tot}$')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies_0, energies_1, energies_2
    del energyarray
    del data

def plot_energy_distribution_total(tensor, energyarray, fp, save_it=True, epoch_nr=None,
                                   use_log=False, lower_threshold=0.):
    """ plots the energy deposited in all layers for a given batch,
        normalized to Etot (total Energy)
        tensor (torch.tensor): input data of shape (nbatch, 504)
        energyarray (torch.tensor): batch of input Etot (nbatch,)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        epoch_nr (int): if not None, epoch nr will be shown in plot
        use_log (bool): if True, axes are in log scale
        lower_threshold (float): threshold to be applied to data before plotting
    """
    energyarray = energyarray.to('cpu').numpy().squeeze(axis=-1)
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    energies = data.sum(axis=-1)

    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    plt.scatter(energyarray, energies / (1e3*energyarray), s=0.5)
    plt.plot([0., 100.], [1., 1.], linestyle='dashed', color='k', linewidth=1.)

    if epoch_nr is not None:
        plt.text(90., 0.9, 'epoch '+str(epoch_nr))
    if use_log:
        plt.yscale('log')

    plt.xlabel(r'$E_\mathrm{tot}$')
    plt.ylabel(r'$\hat{E}_\mathrm{tot} /E_\mathrm{tot}$')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies, energyarray
    del data

def plot_layer_E_ratio(tensor, fp, layer, save_it=True, plot_ref=None, epoch_nr=None,
                       lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the ratio of the difference of the 2 brightest voxels to their sum
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = data.reshape(data.shape[0], -1)
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    top2 = np.sort(data, axis=1)[:, -2:]
    E_ratio = ((top2[:, 1] - top2[:, 0]) / (top2[:, 0] + top2[:, 1]))[top2[:, 1] > 0]

    bins=np.linspace(0, 1, 100)
    if layer == 0:
        axis_label = r'$E_{\mathrm{ratio},0}$'
    elif layer == 1:
        axis_label = r'$E_{\mathrm{ratio},1}$'
    elif layer == 2:
        axis_label = r'$E_{\mathrm{ratio},2}$'
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference_1 = reference_data.loc[plot_ref]\
                ['E_1_layer_{}_{:1.0e}'.format(layer, ref_thres)]
            reference_2 = reference_data.loc[plot_ref]\
                ['E_2_layer_{}_{:1.0e}'.format(layer, ref_thres)]
            reference = ((reference_1 - reference_2) / (reference_2 + reference_1))[reference_1 > 0]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
                if layer == 2:
                    leg_loc = 'upper right'
                else:
                    leg_loc = 'upper left'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
                leg_loc = 'lower left'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
                leg_loc = 'upper right'
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference_1 = reference_data.loc[plot_ref]['E_1_layer_{}'.format(layer)]
            reference_2 = reference_data.loc[plot_ref]['E_2_layer_{}'.format(layer)]
            reference = ((reference_1 - reference_2) / (reference_2 + reference_1))[reference_1 > 0]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
                if layer == 2:
                    leg_loc = 'upper right'
                else:
                    leg_loc = 'upper left'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
                leg_loc = 'lower left'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
                leg_loc = 'upper right'
            _ = plt.hist(reference, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(E_ratio, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(1., 10., 'epoch '+str(epoch_nr))
    plt.legend(fontsize=20)

    plt.xlabel(axis_label)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del E_ratio, data

def plot_shower_depth(tensor, fp, save_it=True, plot_ref=None, epoch_nr=None, lower_threshold=0.,
                      ref_thres=None, plot_GAN=None):
    """ plots the depth of the shower, i.e. the layer that has the last energy deposition
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy().squeeze()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))

    maxdepth = 2* (data[:, 432:].sum(axis=-1) != 0)
    maxdepth[maxdepth == 0] = 1* (data[:, 288:432][maxdepth == 0].sum(axis=-1) != 0)

    bins = [0, 1, 2, 3]
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference_0 = reference_data.loc[plot_ref]\
                ['energy_layer_0_{:1.0e}'.format(ref_thres)]
            reference_1 = reference_data.loc[plot_ref]\
                ['energy_layer_1_{:1.0e}'.format(ref_thres)]
            reference_2 = reference_data.loc[plot_ref]\
                ['energy_layer_2_{:1.0e}'.format(ref_thres)]

            ref_depth = 2* (reference_2 != 0)
            ref_depth[ref_depth == 0] = 1* (reference_1[ref_depth == 0] != 0)
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(ref_depth, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference_0 = reference_data.loc[plot_ref]['energy_layer_0']
            reference_1 = reference_data.loc[plot_ref]['energy_layer_1']
            reference_2 = reference_data.loc[plot_ref]['energy_layer_2']

            ref_depth = 2* (reference_2 != 0)
            ref_depth[ref_depth == 0] = 1* (reference_1[ref_depth == 0] != 0)
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(ref_depth, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(maxdepth, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(0.25, 0.6, 'epoch '+str(epoch_nr))

    plt.legend(loc='upper left', fontsize=20)
    plt.xlabel(r'Max Depth $d$ (layer)')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del maxdepth
    if plot_ref is not None:
        del reference_0, reference_1, reference_2, reference_data, ref_depth

def plot_nn(tensor, tensor_energies, fp, layer, num_events, ref_data_path, ref_data_name,
            save_it=True, epoch_nr=None, lower_threshold=0.):
    """ takes num_events events of the GEANT set and plots the nearest neighbors of the
        generated set of events. If num_events == 'fixed', a set of 5 preselected events will
        be used.
        tensor (torch.tensor): input data of shape (nbatch, 504)
        tensor_energies (torch.tensor): Etot of the input data of shape (nbatch, )
        fp (string): filepath including filename with extension to save image
        layer (int or 'all'): which layer (0, 1, 2) to compare. If 'all', full event will be used
        num_events (int or 'fixed'): number of events to be drawn or fixed set will be used
        ref_data_path (str): file path to folder with GEANT data
        ref_data_name (str): filename (=particle type) of the GEANT data
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy().squeeze()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    data_energies = tensor_energies.numpy().squeeze()*1e2
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(data.reshape(data.shape[0], -1))

    dataset = CaloDataset(ref_data_path, ref_data_name, apply_logit=False)
    if layer == 'all':
        layer0 = (dataset[:]['layer_0']*1e5).reshape(len(dataset), -1)
        layer1 = (dataset[:]['layer_1']*1e5).reshape(len(dataset), -1)
        layer2 = (dataset[:]['layer_2']*1e5).reshape(len(dataset), -1)
        real_images = np.concatenate((layer0, layer1, layer2), axis=1)
        del layer0, layer1, layer2
    else:
        real_images = dataset[:]['layer_'+str(layer)]*1e5

    if num_events == 'fixed':
        if 'eplus' in ref_data_name:
            particles = [19046, 52805, 22249, 10208, 11495]
        elif 'gamma' in ref_data_name:
            particles = [6304, 12894, 68071, 33254, 37427]
        elif 'piplus' in ref_data_name:
            particles = [16457, 23640, 67794, 19485, 49980]
        num_events = 5
    else:
        particles = np.random.randint(low=0, high=real_images.shape[0], size=num_events)

    energies = []
    for particle in particles:
        energies.append(dataset[particle]['energy'][0]*1e2)
    sorted_indices = np.argsort(energies)
    particles = [particles[i] for i in sorted_indices]
    energies = [energies[i] for i in sorted_indices]

    if layer == 'all':
        nbrs = []
        for sel in particles:
            nbrs.append(nn.kneighbors(real_images[sel].ravel().reshape(1, -1))[-1])
        layer_data = np.array_split(data, (288, 432), axis=1)
        real_image = np.array_split(real_images, (288, 432), axis=1)
        f, axarr = plt.subplots(6, num_events, figsize=(18, 18))
        plt.subplots_adjust(left=0.075, right=0.825)
        #for layer_id in range(3):
        for i, nbr in enumerate(nbrs):
            def _plot_im(n_row, images):
                layer_nr = n_row//2
                im = axarr[n_row, i].imshow(images,
                                            interpolation='nearest',
                                            norm=LogNorm(vmin=1e-2, vmax=None),
                                            aspect=SIZES[1+(layer_nr*2)] /\
                                            float(SIZES[(0)+(layer_nr*2)]))
                axarr[n_row, i].tick_params(axis='both', which='both', bottom=False, top=False,
                                            left=False, right=False, labelbottom=False,
                                            labelleft=False)
                return im
            cnv = [_plot_im(*content) for content in enumerate([
                real_image[0][particles[i]].reshape(SIZES[0], SIZES[1]),
                layer_data[0][nbr].reshape(SIZES[0], SIZES[1]),
                real_image[1][particles[i]].reshape(SIZES[2], SIZES[3]),
                layer_data[1][nbr].reshape(SIZES[2], SIZES[3]),
                real_image[2][particles[i]].reshape(SIZES[4], SIZES[5]),
                layer_data[2][nbr].reshape(SIZES[4], SIZES[5])])]
            dep_real = real_image[0][particles[i]].sum() +\
                real_image[1][particles[i]].sum() + real_image[2][particles[i]].sum()
            dep_flow = layer_data[0][nbr].sum() +\
                layer_data[1][nbr].sum() + layer_data[2][nbr].sum()
            #axarr[5][i].set_xlabel(
            #    r'$E_{\mathrm{GEANT}}=$'+'{:.1f} GeV \n'.format(energies[i])+\
            #    r'$\hat{E}_{\mathrm{GEANT}}=$'+'{:.1f} GeV \n'.format(dep_real/1e3)+\
            #    r'$E_{\mathrm{CaloFlow}}=$'+'{:.1f} GeV \n'.format(data_energies[nbr[0, 0]])+\
            #    r'$\hat{E}_{\mathrm{CaloFlow}}=$'+'{:.1f} GeV \n'.format(dep_flow/1e3),
            #    fontsize=20)
            axarr[0][i].set_title(r'$E_{\mathrm{tot}}=$'+'{:.1f} GeV'.format(energies[i]),
                                  fontsize=20)
            axarr[5][i].set_xlabel(
                r'$\hat{E}_{\mathrm{GEANT}}=$'+'{:.1f} GeV \n'.format(dep_real/1e3)+\
                r'$\hat{E}_{\mathrm{CaloFlow}}=$'+'{:.1f} GeV'.format(dep_flow/1e3),
                fontsize=20)
        for layer_id in range(3):
            axarr[2*layer_id + 0][0].set_ylabel('GEANT', fontsize=28)
            axarr[2*layer_id + 1][0].set_ylabel('CaloFlow', fontsize=28)
        cax0 = f.add_axes([0.86, 0.64, 0.03, 0.24])
        cb0 = f.colorbar(cnv[0], cax=cax0)
        cax1 = f.add_axes([0.86, 0.375, 0.03, 0.24])
        cb1 = f.colorbar(cnv[2], cax=cax1)
        cax2 = f.add_axes([0.86, 0.11, 0.03, 0.24])
        cb2 = f.colorbar(cnv[4], cax=cax2)

        cb1.set_label(r'Energy (MeV)', y=0.5, fontsize=28, x=0.96)
        cb0.ax.tick_params(labelsize=20)
        cb1.ax.tick_params(labelsize=20)
        cb2.ax.tick_params(labelsize=20)

        f.text(0.025, 0.21, 'layer 2', rotation=90, fontsize=28)
        f.text(0.025, 0.47, 'layer 1', rotation=90, fontsize=28)
        f.text(0.025, 0.73, 'layer 0', rotation=90, fontsize=28)
        if save_it:
            plt.savefig(fp, dpi=300)
            plt.close()
        else:
            plt.show()
    else:
        print("New Layout is not implemented if layer != 'all' !")
        f, axarr = plt.subplots(2, num_events, figsize=(15, 6))
        for i, sel in enumerate(particles):
            def _plot_im(n_row, images):
                im = axarr[n_row, i].imshow(images,
                                            interpolation='nearest',
                                            norm=LogNorm(vmin=1e-2, vmax=None),
                                            aspect=SIZES[1+(layer*2)] / float(SIZES[(0)+(layer*2)]))
                axarr[n_row, i].tick_params(axis='both', which='both', bottom=False, top=False,
                                            left=False, right=False, labelbottom=False,
                                            labelleft=False)
                return im
            nbr = int(nn.kneighbors(real_images[sel].ravel().reshape(1, -1))[-1])
            cnv = [_plot_im(*content) for content in enumerate([
                real_images[sel].reshape(SIZES[2*layer], SIZES[2*layer +1]),
                data[nbr].reshape(SIZES[2*layer], SIZES[2*layer +1])])]
        axarr[0][0].set_ylabel('GEANT')
        axarr[1][0].set_ylabel('CaloFlow')
        cax = f.add_axes([0.93, 0.11, 0.03, 0.8])
        cb = f.colorbar(cnv[0], cax=cax)
        cb.set_label(r'Energy (MeV)', y=0.73)

        if save_it:
            plt.savefig(fp, dpi=300)
            plt.close()
        else:
            plt.show()

    del dataset, data, nn, real_images

def plot_brightest_voxel(tensor, fp, layer, which_voxel=1, save_it=True, plot_ref=None,
                         epoch_nr=None, lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the ratio of the which_voxel brightest voxels to the energy
        deposited in the layer
        tensor (torch.tensor): input data of shape (nbatch, 288 or 144 or 72)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        which_voxel (int): which voxel in (1, 2, 3, 4, 5) to plot (1=brightest, 5=5th brightest)
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = data.reshape(data.shape[0], -1)
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    top = np.sort(data, axis=1)[:, -which_voxel:]
    energies = data.sum(axis=(1))

    bins = np.linspace(0, 1./which_voxel, 100)

    label = r'''$E_{\mathrm{{%(which_id)d}. brightest}, \mathrm{layer } %(layer_id)d}$'''
    axis_label = label % {'which_id': which_voxel, 'layer_id': layer}

    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus'] and which_voxel in [1, 2, 3, 4, 5]:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference = reference_data.loc[plot_ref]\
                ['E_{}_layer_{}_{:1.0e}'.format(which_voxel, layer, ref_thres)]
            reference_tot = reference_data.loc[plot_ref]\
                ['energy_layer_{}_{:1.0e}'.format(layer, ref_thres)]
            reference = (reference/reference_tot)[reference_tot > 0]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
                if layer == 2:
                    leg_loc = 'upper right'
                else:
                    leg_loc = 'upper left'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
                leg_loc = 'lower left'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
                leg_loc = 'upper right'
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name, ".format(plot_ref) +\
                             "or '{}' is not a voxel on record".format(which_voxel)
                            )
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus'] and which_voxel in [1, 2, 3, 4, 5]:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['E_{}_layer_{}'.format(which_voxel, layer)]
            reference_tot = reference_data.loc[plot_ref]['energy_layer_{}'.format(layer)]
            reference = (reference/reference_tot)[reference_tot > 0]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
                if layer == 2:
                    leg_loc = 'upper right'
                else:
                    leg_loc = 'upper left'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
                leg_loc = 'lower left'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
                leg_loc = 'upper right'
            _ = plt.hist(reference, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name, ".format(plot_GAN) +\
                             "or '{}' is not a voxel on record".format(which_voxel)
                            )

    ratio = (top[:, -which_voxel] / energies)[energies > 0]

    _ = plt.hist(ratio, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(0., 1.75, 'epoch '+str(epoch_nr))
    plt.legend(fontsize=20)
    plt.xlabel(axis_label)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del ratio, data

def plot_depth_weighted_total_energy(tensor, fp, save_it=True, plot_ref=None, epoch_nr=None,
                                     lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the depth-weighted total energy deposit in all 3 layers for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    energies = data[..., 288:432].sum(axis=-1) + 2.* data[..., 432:].sum(axis=-1)
    bins = np.logspace(1, 5, 100)
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference = reference_data.loc[plot_ref]\
                ['energy_layer_1_{:1.0e}'.format(ref_thres)]
            reference += 2.*reference_data.loc[plot_ref]\
                ['energy_layer_2_{:1.0e}'.format(ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['energy_layer_1']
            reference += 2.*reference_data.loc[plot_ref]['energy_layer_2']
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(energies, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(10., 1e-4, 'epoch '+str(epoch_nr))
    plt.ylim([1e-8, 1e-2])
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='upper right', fontsize=20)

    plt.xlabel(r'Lateral Depth $l_d$')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies, data

def plot_depth_weighted_energy_normed(tensor, fp, save_it=True, plot_ref=None, epoch_nr=None,
                                      lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the depth-weighted total energy deposit in all 3 layers
        normalized by the total deposited energy for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    energies = data[..., 288:432].sum(axis=-1) + 2.* data[..., 432:].sum(axis=-1)
    energies /= data.sum(axis=-1)
    bins = np.linspace(0.4, 2., 100)
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference = reference_data.loc[plot_ref]\
                ['energy_layer_1_{:1.0e}'.format(ref_thres)]
            reference += 2.*reference_data.loc[plot_ref]\
                ['energy_layer_2_{:1.0e}'.format(ref_thres)]
            reference /= (reference_data.loc[plot_ref]\
                          ['energy_layer_0_{:1.0e}'.format(ref_thres)]+
                          reference_data.loc[plot_ref]\
                          ['energy_layer_1_{:1.0e}'.format(ref_thres)]+
                          reference_data.loc[plot_ref]\
                          ['energy_layer_2_{:1.0e}'.format(ref_thres)])
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['energy_layer_1']
            reference += 2.*reference_data.loc[plot_ref]['energy_layer_2']
            reference /= (reference_data.loc[plot_ref]['energy_layer_0']+
                          reference_data.loc[plot_ref]['energy_layer_1']+
                          reference_data.loc[plot_ref]['energy_layer_2'])
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(energies, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(1.7, 5., 'epoch '+str(epoch_nr))
    plt.ylim([0., 7.])
    plt.legend(loc='upper right', fontsize=20)

    plt.xlabel(r'Shower Depth $s_d$')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies, data

def plot_depth_weighted_energy_normed_std(tensor, fp, save_it=True, plot_ref=None, epoch_nr=None,
                                          lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the standard deviation of the depth-weighted total energy deposit in all 3 layers
        normalized by the total deposited energy for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    assert data.shape[-1] == 288 + 144 + 72, (
        "Are you sure the input of shape {} is from cells of all 3 layers?".format(tensor.size()))
    layer1 = data[..., 288:432].sum(axis=-1)
    layer2 = data[..., 432:].sum(axis=-1)
    energies = data.sum(axis=-1)
    bins = np.linspace(0., 0.9, 100)
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference1 = reference_data.loc[plot_ref]\
                ['energy_layer_1_{:1.0e}'.format(ref_thres)]
            reference2 = reference_data.loc[plot_ref]\
                ['energy_layer_2_{:1.0e}'.format(ref_thres)]
            referencetot = reference1 + reference2 + reference_data.loc[plot_ref]\
                ['energy_layer_0_{:1.0e}'.format(ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(layer_std(reference1, reference2, referencetot), bins=bins,
                         histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference1 = reference_data.loc[plot_ref]['energy_layer_1']
            reference2 = reference_data.loc[plot_ref]['energy_layer_2']
            referencetot = reference1 + reference2 + reference_data.loc[plot_ref]['energy_layer_0']
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(layer_std(reference1, reference2, referencetot), bins=bins,
                         histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(layer_std(layer1, layer2, energies), bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(0.8, 5., 'epoch '+str(epoch_nr))
    plt.ylim([0., 7.])
    plt.legend(loc='upper right', fontsize=20)

    plt.xlabel(r'Shower Depth Width $\sigma_{s_d}$')

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies, data, layer1, layer2

def plot_layer_lateral_width(tensor, fp, layer, save_it=True, plot_ref=None, epoch_nr=None,
                             lower_threshold=0., ref_thres=None, plot_GAN=None):
    """ plots the standard deviation of the transverse energy profile per layer,
        in units of cell numbers for a given batch
        tensor (torch.tensor): input data of shape (nbatch, 288 or 144 or 72)
        fp (string): filepath including filename with extension to save image
        layer (int): which layer (0, 1, 2) the data is from
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    energies = data.sum(axis=(1, 2))
    eta_cells = {'layer_0' : 3, 'layer_1' : 12, 'layer_2' : 12}
    eta_bins = np.linspace(-240, 240, eta_cells['layer_' + str(layer)] + 1)
    bin_centers = (eta_bins[1:] + eta_bins[:-1]) / 2.

    x = (data * bin_centers.reshape(-1, 1)).sum(axis=(1, 2))
    x2 = (data * (bin_centers.reshape(-1, 1) ** 2)).sum(axis=(1, 2))
    dist = np.sqrt((x2 / (energies+1e-8)) - (x / (energies+1e-8)) ** 2)

    if layer == 0:
        bins = np.logspace(0, 3, 100)
        x_max = 200
        y_max = 5
        axis_label = r'$\sigma_0$'
    elif layer == 1:
        bins = np.logspace(0, 2, 100)
        x_max = 200
        y_max = 2
        axis_label = r'$\sigma_1$'
    elif layer == 2:
        bins = np.logspace(0, 3, 100)
        x_max = 300
        y_max = 2
        axis_label = r'$\sigma_2$'
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference = reference_data.loc[plot_ref]\
                ['layer_{}_lateral_width_{:1.0e}'.format(str(layer), ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference = reference_data.loc[plot_ref]['layer_{}_lateral_width'.format(str(layer))]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(dist, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(0.9, 0.1, 'epoch '+str(epoch_nr))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(xmax=x_max)
    plt.ylim(ymax=y_max)
    plt.legend(loc='upper left', fontsize=20)

    plt.xlabel(axis_label)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies, dist, x, x2

def plot_centroid_correlation(tensor, fp, layer1, layer2, scan='phi', save_it=True,
                              plot_ref=None, epoch_nr=None, lower_threshold=0., ref_thres=None,
                              plot_GAN=None):
    """ plots the difference between the eta/phi positions of the centroid of layer1 and layer2
        tensor (torch.tensor): input data of shape (nbatch, 504)
        fp (string): filepath including filename with extension to save image
        layer1 (int): which layer (0, 1, 2) of the data to compare
        layer2 (int): which layer (0, 1, 2) of the data to compare
        scan (str): 'phi' or 'eta', which direction to compute the centroid in
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
        plot_ref (str): if not None, plot GEANT4 reference of that type (eplus, gamma, piplus)
        epoch_nr (int): if not None, epoch nr will be shown in plot
        lower_threshold (float): threshold to be applied to data before plotting
        ref_thres (float): threshold to be applied to GEANT4 reference data
        plot_GAN (bool): if True, results from our CaloGAN will be plotted, too
    """
    data = tensor.clamp_(0., 1e5).to('cpu').numpy()
    data = np.where(data < lower_threshold, np.zeros_like(data), data)
    if scan == 'phi': # pos along y (phi) axis
        cells = {'layer_0' : 3, 'layer_1' : 12, 'layer_2' : 12}
    elif scan == 'eta': # pos along x (eta) axis
        cells = {'layer_0' : 96, 'layer_1' : 12, 'layer_2' : 6}
    else:
        raise ValueError("scan={} not in ['eta', 'phi']".format(scan))

    if layer2 < layer1:
        TEMP = layer2
        layer2 = layer1
        layer1 = TEMP

    bins1 = np.linspace(-240, 240, cells['layer_' + str(layer1)] + 1)
    bin_centers1 = (bins1[1:] + bins1[:-1]) / 2.
    bins2 = np.linspace(-240, 240, cells['layer_' + str(layer2)] + 1)
    bin_centers2 = (bins2[1:] + bins2[:-1]) / 2.
    layer_pos = {'layer_0' : (0, 288), 'layer_1' : (288, 432), 'layer_2' : (432, 504)}
    start1, end1 = layer_pos['layer_' + str(layer1)]
    start2, end2 = layer_pos['layer_' + str(layer2)]

    if scan == 'phi':
        data1 = data[..., start1:end1].reshape(len(data), cells['layer_' + str(layer1)], -1)
        data2 = data[..., start2:end2].reshape(len(data), cells['layer_' + str(layer2)], -1)
        x1 = (data1 * bin_centers1.reshape(-1, 1)).sum(axis=(1, 2))
        x2 = (data2 * bin_centers2.reshape(-1, 1)).sum(axis=(1, 2))
    else:
        data1 = data[..., start1:end1].reshape(len(data), -1, cells['layer_' + str(layer1)])
        data2 = data[..., start2:end2].reshape(len(data), -1, cells['layer_' + str(layer2)])
        x1 = (data1 * bin_centers1.reshape(1, -1)).sum(axis=(1, 2))
        x2 = (data2 * bin_centers2.reshape(1, -1)).sum(axis=(1, 2))

    energies1 = data1.sum(axis=(1, 2))
    energies2 = data2.sum(axis=(1, 2))
    x1 /= (energies1 + 1e-8)
    x2 /= (energies2 + 1e-8)

    dist = x1-x2

    label = r'''$\langle \%(scan)s_{%(layer1)d}\rangle - \langle \%(scan)s_{%(layer2)d}\rangle$'''
    axis_label = label % {'scan': scan, 'layer1': layer1, 'layer2': layer2}

    bins = np.linspace(-120, 120, 50)
    colors = matplotlib.cm.gnuplot2(np.linspace(0.2, 0.8, 3))
    plt.figure(figsize=(6, 6))

    if plot_ref is not None:
        if ref_thres is None:
            ref_thres = lower_threshold
        if plot_ref in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_{:1.0e}.hdf'.format(ref_thres))
            reference1 = reference_data.loc[plot_ref]\
                ['layer_{}_centroid_{}_{:1.0e}'.format(str(layer1), str(scan), ref_thres)]
            reference2 = reference_data.loc[plot_ref]\
                ['layer_{}_centroid_{}_{:1.0e}'.format(str(layer2), str(scan), ref_thres)]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ GEANT'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ GEANT'
            else:
                color = colors[2]
                label = r'$\pi^+$ GEANT'
            _ = plt.hist(reference1-reference2, bins=bins, histtype='stepfilled',
                         linewidth=2, alpha=0.2, density=True, color=color,
                         label=label)
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_ref))
    if plot_GAN is not None:
        if plot_GAN in ['eplus', 'gamma', 'piplus']:
            reference_data = pd.read_hdf('plots_reference_CaloGAN.hdf')
            reference1 = reference_data.loc[plot_ref]\
                ['layer_{}_centroid_{}'.format(str(layer1), str(scan))]
            reference2 = reference_data.loc[plot_ref]\
                ['layer_{}_centroid_{}'.format(str(layer2), str(scan))]
            if plot_ref == 'eplus':
                color = colors[0]
                label = r'$e^+$ CaloGAN'
            elif plot_ref == 'gamma':
                color = colors[1]
                label = r'$\gamma$ CaloGAN'
            else:
                color = colors[2]
                label = r'$\pi^+$ CaloGAN'
            _ = plt.hist(reference1-reference2, bins=bins, histtype='step',
                         linewidth=2, alpha=0.5, density=True, color=color,
                         label=label, linestyle='dashed')
        else:
            raise ValueError("'{}' is not a valid reference name".format(plot_GAN))

    _ = plt.hist(dist, bins=bins, histtype='step', linewidth=3,
                 alpha=1, color=color if plot_ref is not None else colors[0], density='True',
                 label='CaloFlow')
    if epoch_nr is not None:
        plt.text(20., 0.1, 'epoch '+str(epoch_nr))
    plt.yscale('log')
    plt.legend(loc='upper left', fontsize=20)

    plt.xlabel(axis_label)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp, dpi=300)
        plt.close()
    else:
        plt.show()
    del energies1, energies2, dist, x1, x2, data1, data2

def plot_loss(train_loss, test_loss, fp, save_it=True):
    """ plots the training and test loss
        train_loss: list of losses obtained during training (datasize/batchsize per epoch)
        test_loss: list of losses from the test set (one per epoch)
        fp (list of str): file paths to save plot and loss arrays
        save_it (bool): if True, plot will be saved to fp, if not, plot is only shown
    """
    # test_loss is written once per epoch: gives number of epochs to plot
    num_epochs = len(test_loss)
    num_batches = len(train_loss) / num_epochs
    plt.figure(figsize=(6, 6))
    plt.xlim([0., num_epochs])
    ylim = np.min([np.min(np.array(train_loss)), np.min(np.array(test_loss))])
    plt.ylim([ylim, 1000.])

    test_arg = np.arange(1., num_epochs+1, 1)
    train_arg = np.arange(0., num_epochs, 1./num_batches) + (1./num_batches)

    plt.plot(train_arg, train_loss, 'b', label='training', lw=2.)
    plt.plot(test_arg, test_loss, 'orange', label='test', lw=2.)

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right', fontsize=20)

    plt.tight_layout()
    if save_it:
        plt.savefig(fp[0], dpi=300)
        np.save(fp[1], test_loss)
        np.save(fp[2], train_loss)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':

    # generate random data and test plot routines above
    local_thresholds = [0., 1e-1]
    nrow = 4
    ncol = 6
    npts_1 = nrow * ncol
    print("Testing plot_calo_batch                                                    ", end='\r')
    for layer in [0, 1, 2]:
        x_len = SIZES[2*layer + 1]
        y_len = SIZES[2*layer]

        pts_1 = 10**((torch.rand(size=(npts_1, y_len, x_len))-(4./7.))*7.)
        pts_1 = torch.where(pts_1 < 1e-2, torch.zeros_like(pts_1), pts_1)
        for thres in local_thresholds:
            plot_calo_batch(pts_1, 'dummy', layer, ncol=ncol, save_it=False,
                            lower_threshold=thres)

    print("Testing plot_average_voxel                                                 ", end='\r')
    npts_2 = 10
    for layer in [0, 1, 2]:
        x_len = SIZES[2*layer + 1]
        y_len = SIZES[2*layer]

        pts_2 = 10**((torch.rand(size=(npts_2, y_len, x_len))-(4./7.))*7.)
        pts_2 = torch.where(pts_2 < 1e-2, torch.zeros_like(pts_2), pts_2)
        for thres in local_thresholds:
            plot_average_voxel(pts_2, 'dummy', layer, vmin=None, save_it=False,
                               lower_threshold=thres)

    print("Testing plot_layer_energy                                                  ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    npts_3 = 100
    for layer in [0, 1, 2]:
        x_len = SIZES[2*layer + 1]
        y_len = SIZES[2*layer]

        pts_3 = 10**((torch.rand(size=(npts_3, y_len, x_len))-(4./7.))*7.)
        pts_3 = torch.where(pts_3 < 1e-2, torch.zeros_like(pts_3), pts_3)
        for thres in local_thresholds:
            plot_layer_energy(pts_3, 'dummy', layer, save_it=False, plot_ref=my_plot_ref,
                              lower_threshold=thres, plot_GAN=my_plot_ref)
    print("Testing plot_loss                                                          ", end='\r')
    num_epochs = 5
    num_batches = 14
    train_loss = 400.*np.random.rand(num_epochs*num_batches)
    test_loss = 400.*np.random.rand(num_epochs)
    plot_loss(train_loss, test_loss, 'dummy', save_it=False)

    print("Testing plot_total_energy                                                  ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    npts_4 = 1000
    pts_4 = 10**((torch.rand(size=(npts_4, 504))-(4./7.))*7.)
    for thres in local_thresholds:
        plot_total_energy(pts_4, 'dummy', save_it=False, plot_ref=my_plot_ref,
                          lower_threshold=thres, plot_GAN=my_plot_ref)

    print("Testing plot_energy_fraction                                               ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for use_log in [True, False]:
        for layer in [0, 1, 2]:
            for thres in local_thresholds:
                plot_energy_fraction(pts_4, 'dummy', layer, save_it=False,
                                     plot_ref=my_plot_ref, use_log=use_log,
                                     lower_threshold=thres, plot_GAN=my_plot_ref)

    print("Testing plot_layer_sparsity                                                ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    npts_5 = 1000
    for layer in [0, 1, 2]:
        x_len = SIZES[2*layer + 1]
        y_len = SIZES[2*layer]

        pts_5 = 10**((torch.rand(size=(npts_5, y_len, x_len))-(4./7.))*7.)
        for _ in range(10000):
            pts_5[np.random.randint(npts_5), np.random.randint(y_len), np.random.randint(x_len)]=0.

        for thres in local_thresholds:
            plot_layer_sparsity(pts_5, 'dummy', layer, save_it=False, plot_ref=my_plot_ref,
                                threshold=thres, plot_GAN=my_plot_ref)

    print("Testing plot_energy_distribution 2x                                         ", end='\r')
    npts_6 = 1000
    pts_6 = 10**((torch.rand(size=(npts_6, 504))-(4./7.))*7.)
    eng_6 = (pts_6.sum(axis=-1) + torch.rand(size=(npts_6,))*1e3) / 1e3
    for thres in local_thresholds:
        for use_log in [True, False]:
            plot_energy_distribution_layer(pts_6, eng_6.unsqueeze(-1), 'dummy', save_it=False,
                                           use_log=use_log, lower_threshold=thres)
            plot_energy_distribution_total(pts_6, eng_6.unsqueeze(-1), 'dummy', save_it=False,
                                           use_log=use_log, lower_threshold=thres)

    print("Testing plot_layer_E_ratio                                                  ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for thres in local_thresholds:
        plot_layer_E_ratio(pts_6[..., :288], 'dummy', 0, save_it=False,
                           plot_ref=my_plot_ref, lower_threshold=thres, plot_GAN=my_plot_ref)
        plot_layer_E_ratio(pts_6[..., 288:432], 'dummy', 1, save_it=False,
                           plot_ref=my_plot_ref, lower_threshold=thres, plot_GAN=my_plot_ref)
        plot_layer_E_ratio(pts_6[..., 432:], 'dummy', 2, save_it=False,
                           plot_ref=my_plot_ref, lower_threshold=thres, plot_GAN=my_plot_ref)

    print("Testing plot_shower_depth                                                   ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for thres in local_thresholds:
        plot_shower_depth(pts_6, 'dummy', save_it=False, plot_ref=my_plot_ref,
                          lower_threshold=thres, plot_GAN=my_plot_ref)

    print("Testing plot_nn                                                             ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for thres in local_thresholds:
        plot_nn(pts_6, eng_6, 'dummy', 'all', 5, '/media/claudius/8491-9E93/ML_sources/CaloGAN/',
                my_plot_ref, save_it=False, lower_threshold=thres)
        pts_6_list = torch.split(pts_6, [288, 144, 72], dim=1)
        for layer_id in [0, 1, 2]:
            plot_nn(pts_6_list[layer_id], eng_6, 'dummy', layer_id, 5,
                    '/media/claudius/8491-9E93/ML_sources/CaloGAN/',
                    my_plot_ref, save_it=False, lower_threshold=thres)

    print("Testing plot_brightest_voxel                                                ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for thres in local_thresholds:
        for layer_id in [0, 1, 2]:
            for which_voxel in [1, 2, 3, 4, 5]:
                plot_brightest_voxel(pts_6, 'dummy', layer_id, which_voxel=which_voxel,
                                     save_it=False, plot_ref=my_plot_ref,
                                     lower_threshold=thres, plot_GAN=my_plot_ref)

    print("Testing plot_depth_weighted_total_energy                                    ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for thres in local_thresholds:
        plot_depth_weighted_total_energy(pts_6, 'dummy', save_it=False,
                                         plot_ref=my_plot_ref, lower_threshold=thres,
                                         plot_GAN=my_plot_ref)

    print("Testing plot_centroid_correlation                                           ", end='\r')
    my_plot_ref = np.random.choice(['eplus', 'gamma', 'piplus'])
    for thres in local_thresholds:
        for scan_dir in ['eta', 'phi']:
            plot_centroid_correlation(pts_6, 'dummy', 0, 1, scan=scan_dir, save_it=False,
                                      plot_ref=my_plot_ref, lower_threshold=thres)
            plot_centroid_correlation(pts_6, 'dummy', 0, 2, scan=scan_dir, save_it=False,
                                      plot_ref=my_plot_ref, lower_threshold=thres)
            plot_centroid_correlation(pts_6, 'dummy', 1, 2, scan=scan_dir, save_it=False,
                                      plot_ref=my_plot_ref, lower_threshold=thres)
