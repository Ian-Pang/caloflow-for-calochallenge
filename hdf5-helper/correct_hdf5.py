""" takes one of the first files written by CaloFlow and corrects the dimensions
    of the 'energy' and 'overflow' datasets

    not needed anymore

    "CaloFlow: Fast and Accurate Generation of Calorimeter Showers with Normalizing Flows"
    by Claudius Krause and David Shih
    arxiv:2106.05285

"""

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', help='file to re-write')

args = parser.parse_args()

input_file = h5py.File(args.file, 'r+')

energy = input_file['energy'][:]
del input_file['energy']
input_file.create_dataset('energy', data=energy.reshape(-1, 1)*1e2)

overflow = input_file['overflow'][:]
del input_file['overflow']
input_file.create_dataset('overflow', data=np.zeros((len(overflow), 3)))

input_file.close()
