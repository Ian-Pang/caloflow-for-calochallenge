""" Dataloader for calorimeter data.
    Inspired by https://github.com/kamenbliznashki/normalizing_flows

    Used for
    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

import os
import h5py
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

ALPHA = 1e-6
def logit(x):
    return np.log(x / (1.0 - x))

def logit_trafo(x):
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)


class CaloDataset(Dataset):
    """CaloGAN dataset of [2]."""

    def __init__(self, path_to_file, particle_type,
                 transform_0=None, transform_1=None, transform_2=None, transform_3=None, transform_4=None,
                 apply_logit=True, prefix=None, with_noise=False,
                 return_label=False):
        """
        Args:
            path_to_file (string): path to folder of .hdf5 files
            particle_type (string): name of particle: gamma, eplus, or piplus
            transform_i (callable, optional): Optional transform to be applied
            on data of layer i
        """

        if prefix:
            self.path_to_file = os.path.join(path_to_file, prefix+particle_type+'.hdf5')
        else:
            self.path_to_file = os.path.join(path_to_file, particle_type+'.hdf5')

        self.full_file = h5py.File(self.path_to_file, 'r')

        self.apply_logit = apply_logit
        self.with_noise = with_noise

        self.transform_0 = transform_0
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform_3 = transform_3
        self.transform_4 = transform_4

        self.return_label = return_label

        self.input_dims = {'0': (1,8), '1': (10,16), '2': (10,19), '3': (1,5), '4': (1,5)}
        self.input_size = {'0': 8, '1': 160, '2': 190, '3': 5, '4': 5}

        # normalizations to 100 GeV
        self.file_layer_0 = self.full_file['layer_0'][:] / 1e5
        self.file_layer_1 = self.full_file['layer_1'][:] / 1e5
        self.file_layer_2 = self.full_file['layer_2'][:] / 1e5
        self.file_layer_3 = self.full_file['layer_3'][:] / 1e5
        self.file_layer_4 = self.full_file['layer_4'][:] / 1e5
        self.file_energy = self.full_file['energy'][:] /1e5
        
        if self.return_label:
            self.file_label = self.full_file['label'][:]
        self.full_file.close()

    def __len__(self):
        # assuming file was written correctly
        #return len(self.full_file['energy'])
        return len(self.file_energy)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## normalizations to 100 GeV
        #layer_0 = self.full_file['layer_0'][idx] / 1e5
        #layer_1 = self.full_file['layer_1'][idx] /1e5
        #layer_2 = self.full_file['layer_2'][idx] /1e5
        #energy = self.full_file['energy'][idx] /1e2
        #overflow = self.full_file['overflow'][idx] /1e5
        layer_0 = self.file_layer_0[idx]
        layer_1 = self.file_layer_1[idx]
        layer_2 = self.file_layer_2[idx]
        layer_3 = self.file_layer_3[idx]
        layer_4 = self.file_layer_4[idx]
        energy = self.file_energy[idx]

        if self.with_noise:
            layer_0 = add_noise(layer_0)
            layer_1 = add_noise(layer_1)
            layer_2 = add_noise(layer_2)
            layer_3 = add_noise(layer_3)
            layer_4 = add_noise(layer_4)

        layer_0_E = layer_0.sum(keepdims=True)
        layer_1_E = layer_1.sum(keepdims=True)
        layer_2_E = layer_2.sum(keepdims=True)
        layer_3_E = layer_3.sum(keepdims=True)
        layer_4_E = layer_4.sum(keepdims=True)

        if self.transform_0:
            if self.transform_0 == 'E_norm':
                layer_0 = layer_0 / energy
            elif self.transform_0 == 'L_norm':
                layer_0 = layer_0 / (layer_0_E + 1e-16)
            else:
                layer_0 = self.transform_0(layer_0)

        if self.transform_1:
            if self.transform_1 == 'E_norm':
                layer_1 = layer_1 / energy
            elif self.transform_1 == 'L_norm':
                layer_1 = layer_1 / (layer_1_E + 1e-16)
            else:
                layer_1 = self.transform_1(layer_1)

        if self.transform_2:
            if self.transform_2 == 'E_norm':
                layer_2 = layer_2 / energy
            elif self.transform_2 == 'L_norm':
                layer_2 = layer_2 / (layer_2_E + 1e-16)
            else:
                layer_2 = self.transform_2(layer_2)

        if self.transform_3:
            if self.transform_3 == 'E_norm':
                layer_3 = layer_3 / energy
            elif self.transform_3 == 'L_norm':
                layer_3 = layer_3 / (layer_3_E + 1e-16)
            else:
                layer_3 = self.transform_3(layer_3)
                
        if self.transform_4:
            if self.transform_4 == 'E_norm':
                layer_4 = layer_4 / energy
            elif self.transform_4 == 'L_norm':
                layer_4 = layer_4 / (layer_4_E + 1e-16)
            else:
                layer_4 = self.transform_4(layer_4)

        if self.apply_logit:
            layer_0 = logit_trafo(layer_0)
            layer_1 = logit_trafo(layer_1)
            layer_2 = logit_trafo(layer_2)
            layer_3 = logit_trafo(layer_3)
            layer_4 = logit_trafo(layer_4)

        sample = {'layer_0': layer_0, 'layer_1': layer_1,
                  'layer_2': layer_2, 'layer_3': layer_3, 'layer_4': layer_4, 'energy': energy,
                  'layer_0_E': layer_0_E.squeeze(), 'layer_1_E': layer_1_E.squeeze(), 'layer_2_E': layer_2_E.squeeze(), 'layer_3_E': layer_3_E.squeeze(), 'layer_4_E': layer_4_E.squeeze()}
        if self.return_label:
            #sample['label'] = self.full_file['label'][idx]
            sample['label'] = self.file_label[idx]

        return sample

def get_dataloader(particle_type, data_dir, device, full,
                   batch_size=32, apply_logit=True, with_noise=False, normed=False,
                   normed_layer=False, return_label=False):

    if normed and normed_layer:
        raise ValueError("Cannot normalize data to layer and event simultaenously")

    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type is 'cuda' else {}

    if normed:
        dataset_kwargs = {'transform_0': 'E_norm',
                          'transform_1': 'E_norm',
                          'transform_2': 'E_norm',
                          'transform_3': 'E_norm',
                          'transform_4': 'E_norm',
                          'with_noise': with_noise}
    elif normed_layer:
        dataset_kwargs = {'transform_0': 'L_norm',
                          'transform_1': 'L_norm',
                          'transform_2': 'L_norm',
                          'transform_3': 'L_norm',
                          'transform_4': 'L_norm',
                          'with_noise': with_noise}
    else:
        dataset_kwargs = {'with_noise': with_noise}

    if full:
        dataset = CaloDataset(data_dir, particle_type, apply_logit=apply_logit,
                              return_label=return_label)
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, **kwargs)
    else:
        train_dataset = CaloDataset(data_dir, particle_type, apply_logit=apply_logit,
                                    prefix='train_', return_label=return_label, **dataset_kwargs)
        test_dataset = CaloDataset(data_dir, particle_type, apply_logit=apply_logit,
                                   prefix='test_', return_label=return_label, **dataset_kwargs)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, **kwargs)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False, **kwargs)
        return train_dataloader, test_dataloader

def add_noise(input_tensor):
    noise = np.random.rand(*input_tensor.shape)*1e-8
    #noise = np.random.rand(*input_tensor.shape)*1e-9
    return input_tensor+noise

# make edits to the following (May 25 2022)
def save_samples_to_file(samples, energies, filename, threshold):
    """ saves the given sample to hdf5 file, like training data
        add 0s to overflow to match structure of training data
    """

    assert len(energies) == len(samples)

    data = samples.clamp_(0., 1e6).to('cpu').numpy()
    data = np.where(data < threshold, np.zeros_like(data), data)

    incident_energies = energies.to('cpu').unsqueeze(-1).numpy()*1e5
    showers = data[...,:].reshape(-1,368)

    save_file = h5py.File(filename, 'w')
    save_file.create_dataset('incident_energies', data=incident_energies)
    save_file.create_dataset('showers', data=showers)

    save_file.close()
