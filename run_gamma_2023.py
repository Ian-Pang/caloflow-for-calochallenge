# pylint: disable=invalid-name
""" Main script to run CaloFlow for CaloChallenge Dataset 1 photon (gamma) model.
    code based on https://github.com/bayesiains/nflows and https://arxiv.org/pdf/1906.04032.pdf

    "CaloFlow for CaloChallenge Dataset 1"
    by Claudius Krause, Ian Pang and David Shih
    arxiv:2210.14245

"""

######################################   Imports   #################################################
import argparse
import os
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from nflows import transforms, distributions, flows
from nflows.utils import torchutils
from torch import nn #delete after debugging
import plot_calo_gamma
from data_gamma import get_dataloader
from data_gamma import save_samples_to_file
#from base_dist import ConditionalDiagonalHalfNormal

torch.set_default_dtype(torch.float64)

#####################################   Parser setup   #############################################
parser = argparse.ArgumentParser()

# usage modes
parser.add_argument('--train', action='store_true', help='train a flow')
parser.add_argument('--generate', action='store_true', help='generate from a trained flow and plot')
parser.add_argument('--evaluate', action='store_true', help='evaluate LL of a trained flow')
parser.add_argument('--evaluate_KL', action='store_true',
                    help='evaluate KL of a trained student flow')
parser.add_argument('--generate_to_file', action='store_true',
                    help='generate from a trained flow and save to file')
parser.add_argument('--save_only_weights', action='store_true',
                    help='Loads full model file (incl. optimizer) and saves only weights')
parser.add_argument('--save_every_epoch', action='store_true',
                    help='Saves weights (no optimizer) of every student epoch')

parser.add_argument('--student_mode', action='store_true',
                    help='Work with IAF-student instead of MAF-teacher')

parser.add_argument('--no_cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--which_cuda', default=0, type=int,
                    help='Which cuda device to use')

parser.add_argument('--output_dir', default='./results', help='Where to store the output')
parser.add_argument('--results_file', default='results.txt',
                    help='Filename where to store settings and test results.')
parser.add_argument('--restore_file', type=str, default=None, help='Model file to restore.')
parser.add_argument('--student_restore_file', type=str, default=None,
                    help='Student model file to restore.')
parser.add_argument('--data_dir',help='Where to find the training dataset')

# CALO specific
parser.add_argument('--with_noise', action='store_true',
                    help='Add 1e-8 noise (w.r.t. 100 GeV) to dataset to avoid voxel with 0 energy')
parser.add_argument('--threshold', type=float, default=0.01,
                    help='Threshold in MeV below which voxel energies are set to 0. in plots.')

# MAF parameters
parser.add_argument('--n_blocks', type=int, default=8,
                    help='Total number of blocks to stack in a model (MADE in MAF).')
parser.add_argument('--student_n_blocks', type=int, default=8,
                    help='Total number of blocks to stack in the student model (MADE in IAF).')
parser.add_argument('--hidden_size', type=int, default=378,
                    help='Hidden layer size for each MADE block in an MAF.')
parser.add_argument('--student_hidden_size', type=int, default=378,
                    help='Hidden layer size for each MADE block in the student IAF.')
parser.add_argument('--student_width', type=float, default=1.,
                    help='Width of the base dist. that is used for student training.')
parser.add_argument('--hidden_size_multiplier', type=int, default=None,
                    help='Hidden layer size for each MADE block in an MAF'+\
                    ' is given by the dimension times this factor.')
parser.add_argument('--n_hidden', type=int, default=1,
                    help='Number of hidden layers in each MADE.')
parser.add_argument('--activation_fn', type=str, default='relu',
                    help='What activation function of torch.nn.functional to use in the MADEs.')
parser.add_argument('--batch_norm', action='store_true', default=False,
                    help='Use batch normalization')
parser.add_argument('--n_bins', type=int, default=8,
                    help='Number of bins if piecewise transforms are used')
parser.add_argument('--use_residual', action='store_true', default=False,
                    help='Use residual layers in the NNs')
parser.add_argument('--dropout_probability', '-d', type=float, default=0.05,
                    help='dropout probability')
parser.add_argument('--tail_bound', type=float, default=14., help='Domain of the RQS')
parser.add_argument('--cond_base', action='store_true', default=False,
                    help='Use Gaussians conditioned on energy as base distribution.')
parser.add_argument('--beta', type=float, default=0.5,
                    help='Sets the relative weight between z-chi2 loss (beta=0) and x-chi2 loss')

# training params
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4, help='Initial Learning rate.')
parser.add_argument('--log_interval', type=int, default=175,
                    help='How often to show loss statistics and save samples.')
parser.add_argument('--fully_guided', action='store_true', default=False,
                    help='Train student "fully-guided", ie enable train_xz and train_p')
parser.add_argument('--train_xz', action='store_true', default=False,
                    help="Train student with MSE of all intermediate x's and z's")
parser.add_argument('--train_p', action='store_true', default=False,
                    help='Train student with MSE of all MADE-NN outputs '+\
                    '(to-be parameters of the RQS)')

#######################################   helper functions   #######################################

# used in transformation between energy and logit space:
# (should match the ALPHA in data_mod.py)
ALPHA = 1e-6
def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))

def logit_trafo(x):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = ALPHA + (1. - 2.*ALPHA) * x
    return logit(local_x)

def inverse_logit(x, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((torch.sigmoid(x) - ALPHA) / (1. - 2.*ALPHA)).clamp_(clamp_low, clamp_high)

class IAFRQS(transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform):
    """ IAF version of nflows MAF-RQS"""
    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)
    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)
        
class GuidedCompositeTransform(transforms.CompositeTransform):
    """Composes several transforms into one (in the order they are given),
       optionally returns intermediate results (steps) and NN outputs (p)"""
    
    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__(transforms)
        self._transforms = torch.nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context, direction, return_steps=False, return_p=False):
        steps = [inputs]
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        ret_p = []
        for func in funcs:
            if hasattr(func.__self__, '_transform') and return_p:
                # in student IAF
                if direction == 'forward':
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                else:
                    ret_p.append(func.__self__._transform.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
            elif hasattr(func.__self__, 'autoregressive_net') and return_p:
                # in teacher MAF
                if direction == 'forward':
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
                    outputs, logabsdet = func(outputs, context)
                else:
                    outputs, logabsdet = func(outputs, context)
                    ret_p.append(func.__self__.autoregressive_net(outputs, context))
            else:
                outputs, logabsdet = func(outputs, context)
            steps.append(outputs)
            total_logabsdet += logabsdet
        if return_steps and return_p:
            return outputs, total_logabsdet, steps, ret_p
        elif return_steps:
            return outputs, total_logabsdet, steps
        elif return_p:
            return outputs, total_logabsdet, ret_p
        else:
            return outputs, total_logabsdet
        
    def forward(self, inputs, context=None, return_steps=False, return_p=False):
        #funcs = self._transforms
        funcs = (transform.forward for transform in self._transforms)
        return self._cascade(inputs, funcs, context, direction='forward',
                             return_steps=return_steps, return_p=return_p)

    def inverse(self, inputs, context=None, return_steps=False, return_p=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, direction='inverse',
                             return_steps=return_steps, return_p=return_p)

def remove_nans(tensor):
    """removes elements in the given batch that contain nans
       returns the new tensor and the number of removed elements"""
    tensor_flat = tensor.flatten(start_dim=1)
    good_entries = torch.all(tensor_flat == tensor_flat, axis=1)
    res_flat = tensor_flat[good_entries]
    tensor_shape = list(tensor.size())
    tensor_shape[0] = -1
    res = res_flat.reshape(tensor_shape)
    return res, len(tensor) - len(res)

@torch.no_grad()
def logabsdet_of_base(noise, width=1.):
    """ for computing KL of student"""
    shape = noise.size()[1]
    ret = -0.5 * torchutils.sum_except_batch((noise/width) ** 2, num_batch_dims=1)
    log_z = torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64)
    return ret - log_z

def transform_to_energy(sample, args, scaling=0., Ehat=None):
    """ transforms samples from logit space to energy space, possibly applying a scaling
        with Etot (scaling) and/or Ehat
    """
    sample = ((torch.sigmoid(sample) - ALPHA) / (1. - 2.*ALPHA))

    sample0, sample1, sample2, sample3, sample4= torch.split(sample, args.dim_split, dim=-1)
    sample0 = (sample0 / sample0.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 0].reshape(-1, 1, 1)
    sample1 = (sample1 / sample1.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 1].reshape(-1, 1, 1)
    sample2 = (sample2 / sample2.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 2].reshape(-1, 1, 1)
    sample3 = (sample3 / sample3.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 3].reshape(-1, 1, 1)
    sample4 = (sample4 / sample4.abs().sum(dim=(-1), keepdims=True))\
        * scaling[:, 4].reshape(-1, 1, 1)
    sample = torch.cat((sample0, sample1, sample2, sample3, sample4), 2)
    sample = sample*1e5

    return sample



def split_and_concat(generate_fun, batch_size, model, args, num_pts, energies, rec_model=None):
    """ generates events in batches of size batch_size, if needed """
    starting_time = time.time()
    energy_split = energies.split(batch_size)
    ret = []
    for iteration, energy_entry in enumerate(energy_split):
        ret.append(generate_fun(model, args, num_pts, energy_entry, rec_model).to('cpu'))
        print("Generated {}%".format((iteration+1.)*100. / len(energy_split)), end='\r')
    ending_time = time.time()
    total_time = ending_time - starting_time
    time_string = "Needed {:d} min and {:.1f} s to generate {} events in {} batch(es)."+\
        " This means {:.2f} ms per event."
    print(time_string.format(int(total_time//60), total_time%60, num_pts*len(energies),
                             len(energy_split), total_time*1e3 / (num_pts*len(energies))))
    print(time_string.format(int(total_time//60), total_time%60, num_pts*len(energies),
                             len(energy_split), total_time*1e3 / (num_pts*len(energies))),
          file=open(args.results_file, 'a'))
    return torch.cat(ret)

def generate_to_file(model, args, num_events=9000, energies=None, rec_model=None):
    """ generates samples from the trained model and saves them to file """
    if energies is None:
        energies = 0.99*torch.rand((num_events,)) + 0.01
    scaling = torch.reshape(energies, (-1, 1, 1)).to(args.device)
    samples = split_and_concat(generate_single_with_rec, 10000, model, args, 1, energies,
                                   rec_model)
    filename = os.path.join(args.output_dir, 'CaloFlow_gamma.hdf5')
    save_samples_to_file(samples, energies, filename, args.threshold)

def train_and_evaluate(model, train_loader, test_loader, optimizer, args, rec_model):
    """ As the name says, train the flow and evaluate along the way """
    best_eval_logprob = float('-inf')

    lr_schedule = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                    base_lr=0.5E-4, max_lr=2.0E-3,
                                                    step_size_up=850, step_size_down=None,
                                                    mode='triangular2', gamma=0.99995, scale_fn=None, scale_mode='cycle',
                                                    cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=- 1,
                                                    verbose=False)
    for i in range(args.n_epochs):
        train(model, train_loader, optimizer, i, args, lr_schedule)
        with torch.no_grad():
            eval_logprob, _ = evaluate(model, test_loader, i, args)
            args.test_loss.append(-eval_logprob.to('cpu').numpy())
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            save_all(model, optimizer, args)

def train(model, dataloader, optimizer, epoch, args, lr_schedule):
    """ train the flow one epoch """
    train_single_with_rec(model, dataloader, optimizer, epoch, args, lr_schedule)

@torch.no_grad()
def evaluate(model, dataloader, epoch, args, num_batches=None):
    """Evaluate the model, i.e find the mean log_prob of the test set
       Energy is taken to be the energy of the image, so no
       marginalization is performed.
    """
    return evaluate_single_with_rec(model, dataloader, epoch, args, num_batches=num_batches)

def save_all(model, optimizer, args, is_student=False):
    """ saves the model and the optimizer to file """
    
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                os.path.join(args.output_dir,
                            'student.pt' if is_student else 'model_checkpoint.pt'))

def save_weights(model, args, is_student=False, name=None):
    """ saves the model to file """
    if name is not None:
        file_name = name
    else:
        file_name = 'student_weights.pt' if is_student else 'weight_checkpoint.pt'
    torch.save({'model_state_dict': model.state_dict()},
                os.path.join(args.output_dir, file_name))

def load_all(model, optimizer, args, is_student=False):
    """ loads the model and optimizer from file """
    if is_student:
        filename = args.student_restore_file if args.student_restore_file is not None\
            else 'student.pt'
    else:
        filename = args.restore_file if args.restore_file is not None else 'model_checkpoint.pt'
    #checkpoint = torch.load(os.path.join(args.output_dir, filename))
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(args.device)
    model.eval()

def load_weights(model, args, is_student=False):
    """ loads the model from file """
    if is_student:
        filename = args.student_restore_file if args.student_restore_file is not None\
            else 'student_weights.pt'
    else:
        filename = args.restore_file if args.restore_file is not None else 'weight_checkpoint.pt'
    checkpoint = torch.load(filename, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()

def save_rec_flow(rec_model, args):
    """saves flow that learns energies recursively """
    torch.save({'model_state_dict': rec_model.state_dict()},
               os.path.join('./rec_energy_flow/', 'gamma.pt'))

def load_rec_flow(rec_model, args):
    """ loads flow that learns energies recursively """
    checkpoint = torch.load(os.path.join('./rec_energy_flow/', 'gamma.pt'),
                            map_location='cpu')
    rec_model.load_state_dict(checkpoint['model_state_dict'])
    rec_model.to(args.device)
    rec_model.eval()

def trafo_to_unit_space(energy_array):
    """ transforms energy array to be in [0, 1] """
    num_dim = len(energy_array[0])-2
    ret = [(torch.sum(energy_array[:, :-1], dim=1)/(2.42*energy_array[:, -1])).unsqueeze(1)]
    for n in range(num_dim):
        ret.append((energy_array[:, n]/energy_array[:, n:-1].sum(dim=1)).unsqueeze(1))
    return torch.cat(ret, 1).clamp_(0., 1.)

def trafo_to_energy_space(unit_array, etot_array):
    """ transforms unit array to be back in energy space """
    assert len(unit_array) == len(etot_array)
    num_dim = len(unit_array[0])
    unit_array = torch.cat((unit_array, torch.ones(size=(len(unit_array), 1))), 1)
    ret = [torch.zeros_like(etot_array)]
    ehat_array = unit_array[:, 0] * 2.42 * etot_array
    for n in range(num_dim):
        ret.append(unit_array[:, n+1]*(ehat_array-torch.cat(ret).view(
            n+1, -1).transpose(0, 1).sum(dim=1)))
    ret.append(ehat_array)
    return torch.cat(ret).view(num_dim+2, -1)[1:].transpose(0, 1)

################################# auxilliary NNs and classes #######################################

class RandomPermutationLayer(transforms.Permutation):
    """ Permutes elements with random, but fixed permutation. Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be permuted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.random.permutation(features_entry)
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

class InversionLayer(transforms.Permutation):
    """ Inverts the order of the elements in each layer.  Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be inverted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.arange(features_entry)[::-1]
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

######################## train and evaluation functions for single flow ############################
def train_single_with_rec(model, dataloader, optimizer, epoch, args, lr_schedule):
    """ train recursive single flow one step """
    model.train()
    for i, data in enumerate(dataloader):
        x0 = data['layer_0']
        x1 = data['layer_1']
        x2 = data['layer_2']
        x3 = data['layer_3']
        x4 = data['layer_4']
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']
        E3 = data['layer_3_E']
        E4 = data['layer_4_E']
        E  = data['energy']

        energy = torch.log10(E*3.)
        #energy = torch.log10(E/10.)
        E0 = torch.log10(E0.unsqueeze(-1)+1e-8) - 1.
        E1 = torch.log10(E1.unsqueeze(-1)+1e-8) - 1.
        E2 = torch.log10(E2.unsqueeze(-1)+1e-8) - 1.
        E3 = torch.log10(E3.unsqueeze(-1)+1e-8) - 1.
        E4 = torch.log10(E4.unsqueeze(-1)+1e-8) - 1.


        #y = torch.cat((energy, energy_dists, E0, E1, E2), 1).to(args.device)
        y = torch.cat((energy, E0, E1, E2, E3, E4), 1).to(args.device)

        layer0 = x0.view(x0.shape[0], -1)
        layer1 = x1.view(x1.shape[0], -1)
        layer2 = x2.view(x2.shape[0], -1)
        layer3 = x3.view(x3.shape[0], -1)
        layer4 = x4.view(x4.shape[0], -1)

        x = torch.cat((layer0, layer1, layer2, layer3, layer4), 1).to(args.device)

        loss = - model.log_prob(x, y).mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedule.step()

        args.train_loss.append(loss.tolist())

        if i % args.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, args.n_epochs, i, len(dataloader), loss.item()))
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch+1, args.n_epochs, i, len(dataloader), loss.item()),
                  file=open(args.results_file, 'a'))

@torch.no_grad()
def evaluate_single_with_rec(model, dataloader, epoch, args, num_batches=None):
    """Evaluate the single flow with energy distribution flow, i.e find the mean log_prob
       of the test set. Energy is taken to be the energy of the event, so no
       marginalization is performed.
    """
    model.eval()
    loglike = []
    for batch_id, data in enumerate(dataloader):
        x0 = data['layer_0']
        x1 = data['layer_1']
        x2 = data['layer_2']
        x3 = data['layer_3']
        x4 = data['layer_4']
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']
        E3 = data['layer_3_E']
        E4 = data['layer_4_E']
        E  = data['energy']

        energy = torch.log10(E*3.)
        E0 = torch.log10(E0.unsqueeze(-1)+1e-8) - 1.
        E1 = torch.log10(E1.unsqueeze(-1)+1e-8) - 1.
        E2 = torch.log10(E2.unsqueeze(-1)+1e-8) - 1.
        E3 = torch.log10(E3.unsqueeze(-1)+1e-8) - 1.
        E4 = torch.log10(E4.unsqueeze(-1)+1e-8) - 1.

        y = torch.cat((energy, E0, E1, E2, E3, E4), 1).to(args.device)
        
        layer0 = x0.view(x0.shape[0], -1)
        layer1 = x1.view(x1.shape[0], -1)
        layer2 = x2.view(x2.shape[0], -1)
        layer3 = x3.view(x3.shape[0], -1)
        layer4 = x4.view(x4.shape[0], -1)
        x = torch.cat((layer0, layer1, layer2, layer3, layer4), 1).to(args.device)
            
        loglike.append(model.log_prob(x, y))
        if num_batches is not None:
            if batch_id == num_batches-1:
                break

    logprobs = torch.cat(loglike, dim=0).to(args.device)

    logprob_mean = logprobs.mean(0)
    logprob_std = logprobs.var(0).sqrt()# / np.sqrt(len(dataloader.dataset))

    output = 'Evaluate ' + (epoch is not None)*'(epoch {}) -- '.format(epoch+1) +\
        'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

    print(output.format(logprob_mean, logprob_std))
    print(output.format(logprob_mean, logprob_std), file=open(args.results_file, 'a'))
    return logprob_mean, logprob_std

@torch.no_grad()
def generate_single_with_rec(model, args, num_pts, energies, rec_model):
    """ Generate Samples from single flow with energy flow """
    model.eval()
    energy_dist_unit = sample_rec_flow(rec_model, num_pts, args, energies).to('cpu')
    energy_dist = trafo_to_energy_space(energy_dist_unit, energies)

    energies = torch.log10(energies*3.).unsqueeze(-1)
    E0 = torch.log10(energy_dist[:, 0].unsqueeze(-1)+1e-8) - 1. 
    E1 = torch.log10(energy_dist[:, 1].unsqueeze(-1)+1e-8) - 1. 
    E2 = torch.log10(energy_dist[:, 2].unsqueeze(-1)+1e-8) - 1. 
    E3 = torch.log10(energy_dist[:, 3].unsqueeze(-1)+1e-8) - 1. 
    E4 = torch.log10(energy_dist[:, 4].unsqueeze(-1)+1e-8) - 1.

    y = torch.cat((energies, E0, E1, E2, E3, E4), 1).to(args.device)

    samples = model.sample(num_pts, y)
    samples = transform_to_energy(samples, args, scaling=energy_dist.to(args.device))

    return samples

################## train and evaluation functions for recursive flow ###############################

def train_rec_flow(rec_model, train_data, test_data, optim, args):
    """ trains the flow that learns the energy distributions """
    best_eval_logprob_rec = float('-inf')

    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1.0E-3, total_steps=None, epochs=100, steps_per_epoch=424, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=True, last_epoch=- 1, verbose=False)
    num_epochs = 100 
    for epoch in range(num_epochs):
        rec_model.train()
        for i, data in enumerate(train_data):

            E0 = data['layer_0_E']
            E1 = data['layer_1_E']
            E2 = data['layer_2_E']
            E3 = data['layer_3_E']
            E4 = data['layer_4_E']
            E  = data['energy']

            y = torch.log10(E*3.).to(args.device)

            x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
                                               E1.unsqueeze(1),
                                               E2.unsqueeze(1),
                                               E3.unsqueeze(1),
                                               E4.unsqueeze(1),
                                               E), 1)).to(args.device)
            x = logit_trafo(x)
            loss = - rec_model.log_prob(x, y).mean(0)

            optim.zero_grad()
            loss.backward()
            optim.step()
            lr_schedule.step() #added for onecycle
            if i % args.log_interval == 0:
                print('Recursive Flow: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, i, len(train_data), loss.item()))
                print('Recursive Flow: epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                    epoch+1, num_epochs, i, len(train_data), loss.item()),
                      file=open(args.results_file, 'a'))

        with torch.no_grad():
            rec_model.eval()
            loglike = []
            for data in test_data:
                E0 = data['layer_0_E']
                E1 = data['layer_1_E']
                E2 = data['layer_2_E']
                E3 = data['layer_3_E']
                E4 = data['layer_4_E']
                E  = data['energy']
  
                y = torch.log10(E*3.).to(args.device)
        
                x = trafo_to_unit_space(torch.cat((E0.unsqueeze(1),
                                                   E1.unsqueeze(1),
                                                   E2.unsqueeze(1),
                                                   E3.unsqueeze(1),
                                                   E4.unsqueeze(1),
                                                   E), 1)).to(args.device)

                x = logit_trafo(x)
                loglike.append(rec_model.log_prob(x, y))

            logprobs = torch.cat(loglike, dim=0).to(args.device)

            logprob_mean = logprobs.mean(0)
            logprob_std = logprobs.var(0).sqrt()# / np.sqrt(len(test_data.dataset))

            output = 'Recursive Flow: Evaluate (epoch {}) -- '.format(epoch+1) +\
                'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

            print(output.format(logprob_mean, logprob_std))
            print(output.format(logprob_mean, logprob_std), file=open(args.results_file, 'a'))
            eval_logprob_rec = logprob_mean
        #lr_schedule.step() #commented for onecycle
        if eval_logprob_rec > best_eval_logprob_rec:
            best_eval_logprob_rec = eval_logprob_rec
            save_rec_flow(rec_model, args)

@torch.no_grad()
def sample_rec_flow(rec_model, num_pts, args, energies):
    """ samples layer energies for given total energy from rec flow """
    rec_model.eval()

    context = torch.log10(energies*3.).to(args.device)
    samples = rec_model.sample(num_pts, context.unsqueeze(-1))
    samples = inverse_logit(samples.squeeze())
    return samples

############################ train and evaluate student flow #######################################

def chi2_loss(input1, input2):
    # plain:
    ret = (((input1 - input2)**2).sum(dim=1)).mean()

    return ret

def train_and_evaluate_student(teacher, student, train_loader, test_loader, optimizer_student,
                               args, rec_model=None):
    """ train the student and evaluate it along the way """
    #best_eval_logprob = float('-inf')
    best_eval_KL = float('inf')

    ##initialize energies for student here ######
    list1 = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
    list2 = [1048576, 2097152, 4194304]
    energy_list = []
    for energy in list1:
        energy_list = np.concatenate((energy_list,(np.ones(int(10000))*energy/1e5)))
    count = 4
    energy_list = np.concatenate((energy_list,(np.ones(int(5000))*524288/1e5)))
    for energy in list2:
        count -= 1
        energy_list= np.concatenate((energy_list,(np.ones(int(1000*count))*energy/1e5)))
    energy_list = np.reshape(energy_list, (1,121000))
    student_energies = torch.from_numpy(energy_list)
    student_energies = torch.reshape(student_energies,(-1,))
    student_energies_idx = torch.randperm(student_energies.shape[0]) # shuffle index energies
    student_energies = student_energies[student_energies_idx].view(student_energies.size())  # shuffle energies based on indices 
    student_energies = student_energies[:84700]

    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer_student, max_lr=1.0E-3, total_steps=None, epochs=100, steps_per_epoch=484, pct_start=0.4, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=True, last_epoch=- 1, verbose=False)
    
    with torch.no_grad():
        print("FYI: the teacher LL is:")
        print("FYI: the teacher LL is:", file=open(args.results_file, 'a'))
        eval_logprob_teacher, _ = evaluate(teacher, test_loader, -1, args)

    for i in range(args.n_epochs):
        optimizer_student.zero_grad()
        student_energies_idx = torch.randperm(student_energies.shape[0]) # shuffle index energies after each epoch
        student_energies = student_energies[student_energies_idx].view(student_energies.size())  # shuffle energies based on indices after each epoch
        eval_KL = train_student(teacher, student, train_loader, optimizer_student, i, args, student_energies, lr_schedule)

        args.test_loss.append(eval_KL)
        if eval_KL < best_eval_KL:
            best_eval_KL = eval_KL
            save_all(student, optimizer_student, args, is_student=True)
        if args.save_every_epoch:
            save_weights(student, args, is_student=True, name='student_'+str(i)+'.pt')
        #lr_schedule.step()

def train_student(teacher, student, dataloader, optimizer, epoch, args, student_energies, lr_schedule):
    """ train student with recursive single teacher """
    teacher.eval()
    student.train()

    KL = []

    for i, data in enumerate(dataloader):
        x0 = data['layer_0']
        x1 = data['layer_1']
        x2 = data['layer_2']
        x3 = data['layer_3']
        x4 = data['layer_4']
        E0 = data['layer_0_E']
        E1 = data['layer_1_E']
        E2 = data['layer_2_E']
        E3 = data['layer_3_E']
        E4 = data['layer_4_E']
        E = data['energy']
    
        try:
            z_energies = student_energies[(args.batch_size)*i:(args.batch_size)*(i+1)]
        except:
            z_energies = student_energies[(args.batch_size)*i:]
   
        z_energy_dist_unit = sample_rec_flow(rec_model, 1, args, z_energies).to('cpu')
        z_energy_dist = trafo_to_energy_space(z_energy_dist_unit, z_energies)

        energy = torch.log10(E*3.)
        E0 = torch.log10(E0.unsqueeze(-1)+1e-8) - 1.
        E1 = torch.log10(E1.unsqueeze(-1)+1e-8) - 1.
        E2 = torch.log10(E2.unsqueeze(-1)+1e-8) - 1.
        E3 = torch.log10(E3.unsqueeze(-1)+1e-8) - 1.
        E4 = torch.log10(E4.unsqueeze(-1)+1e-8) - 1.
        z_energies = torch.log10(z_energies*3.).unsqueeze(-1)
        z_E0 = torch.log10(z_energy_dist[:, 0].unsqueeze(-1)+1e-8) - 1.
        z_E1 = torch.log10(z_energy_dist[:, 1].unsqueeze(-1)+1e-8) - 1.
        z_E2 = torch.log10(z_energy_dist[:, 2].unsqueeze(-1)+1e-8) - 1.
        z_E3 = torch.log10(z_energy_dist[:, 3].unsqueeze(-1)+1e-8) - 1.
        z_E4 = torch.log10(z_energy_dist[:, 4].unsqueeze(-1)+1e-8) - 1.


        y = torch.cat((energy, E0, E1, E2, E3, E4), 1).to(args.device)
        z_y = torch.cat((z_energies, z_E0, z_E1, z_E2, z_E3, z_E4), 1).to(args.device)

        layer0 = x0.view(x0.shape[0], -1).to(args.device)
        layer1 = x1.view(x1.shape[0], -1).to(args.device)
        layer2 = x2.view(x2.shape[0], -1).to(args.device)
        layer3 = x3.view(x3.shape[0], -1).to(args.device)
        layer4 = x4.view(x4.shape[0], -1).to(args.device)
        x = torch.cat((layer0, layer1, layer2, layer3, layer4), 1)

        # data MSE (x-space)
        if args.train_xz and args.train_p:
            teacher_noise, _, data_steps_teacher, data_p_teacher = \
                teacher._transform.forward(x, y, return_steps=True, return_p=True)
            student_data, _, data_steps_student, data_p_student = \
                student._transform.inverse(teacher_noise, y, return_steps=True, return_p=True)
        elif args.train_xz:
            teacher_noise, _, data_steps_teacher = teacher._transform.forward(x, y,
                                                                              return_steps=True)
            student_data, _, data_steps_student = student._transform.inverse(teacher_noise, y,
                                                                             return_steps=True)
        elif args.train_p:
            teacher_noise, _, data_p_teacher = teacher._transform.forward(x, y,
                                                                          return_p=True)
            student_data, _, data_p_student = student._transform.inverse(teacher_noise, y,
                                                                         return_p=True)
        else:
            teacher_noise, _ = teacher._transform.forward(x, y)
            student_data, _ = student._transform.inverse(teacher_noise, y)

        loss_chi_x = chi2_loss(student_data, x)

        if args.train_xz:
            for idx, step in enumerate(data_steps_student):
                loss_chi_x += chi2_loss(step, data_steps_teacher[-idx-1])
                #print(chi2_loss(step, data_steps_teacher[-idx-1]))
        if args.train_p:
            for idx, step in enumerate(data_p_student):
                loss_chi_x += chi2_loss(step, data_p_teacher[-idx-1])

        # latent MSE (z-space):
        noise = (torch.randn(args.batch_size, args.dim_sum)*args.student_width).to(args.device)
        if args.train_xz and args.train_p:
            pts, log_student_pre, latent_steps_student, latent_p_student = \
                student._transform.inverse(noise, z_y, return_steps=True, return_p=True)
            latent_teacher_noise, log_teacher_pre, latent_steps_teacher, latent_p_teacher = \
                teacher._transform.forward(pts, z_y, return_steps=True, return_p=True)
        elif args.train_xz:
            pts, log_student_pre, latent_steps_student = student._transform.inverse(
                noise, z_y,
                return_steps=True)
            latent_teacher_noise, log_teacher_pre, latent_steps_teacher = \
                teacher._transform.forward(pts, z_y, return_steps=True)
        elif args.train_p:
            pts, log_student_pre, latent_p_student = student._transform.inverse(
                noise, z_y, return_p=True)
            latent_teacher_noise, log_teacher_pre, latent_p_teacher = \
                teacher._transform.forward(pts, z_y, return_p=True)
        else:
            pts, log_student_pre = student._transform.inverse(noise, z_y)
            latent_teacher_noise, log_teacher_pre = teacher._transform.forward(pts, z_y)

        loss_chi_z = chi2_loss(latent_teacher_noise, noise)
        if args.train_xz:
            for idx, step in enumerate(latent_steps_student):
            #print(idx,step)
                loss_chi_z += chi2_loss(step, latent_steps_teacher[-idx-1])
                #print(chi2_loss(step, data_steps_teacher[-idx-1]))
        if args.train_p:
            for idx, step in enumerate(latent_p_student):
                loss_chi_z += chi2_loss(step, latent_p_teacher[-idx-1])

        with torch.no_grad():
            # KL:
            logabsdet_noise_student = logabsdet_of_base(noise, width=args.student_width)
            logabsdet_noise_teacher = logabsdet_of_base(latent_teacher_noise,
                                                        width=args.student_width)

            # log_of_Sample_And_Log_Prob = logabsdet_of_base + log_of_forward_pass
            # log_of_Sample_And_Log_Prob = logabsdet_of_base - log_of_inverse_pass

            log_teacher = logabsdet_noise_teacher + log_teacher_pre
            log_student = logabsdet_noise_student - log_student_pre

            #KL = (log_student-log_teacher.detach()).mean()
            KL_local = (log_student-log_teacher.detach()).mean()
            KL.append(log_student-log_teacher.detach())
        loss = (1.-args.beta) * loss_chi_z + args.beta * loss_chi_x

        if epoch < 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            optimizer.zero_grad()


        args.train_loss.append(loss.tolist())

        if i % args.log_interval == 0:
            print_string = 'epoch {:3d} / {}, step {:4d} / {}; x-chi {:.4f}; '+\
            'z-chi {:.4f}; loss {:.4f}; KL {:.4f} '
            print(print_string.format(
                epoch+1, args.n_epochs, i, len(dataloader), loss_chi_x.item(), loss_chi_z.item(),
                loss.item(), KL_local.item()))
            print(print_string.format(
                epoch+1, args.n_epochs, i, len(dataloader), loss_chi_x.item(), loss_chi_z.item(),
                loss.item(), KL_local.item()), file=open(args.results_file, 'a'))
    KL_numpy = torch.cat(KL, dim=0).to('cpu').numpy()
    KL_mean = KL_numpy.mean()
    KL_std = KL_numpy.std()
    print("KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))))
    print("KL of epoch is {} +/- {} ({})".format(KL_mean, KL_std, KL_std/np.sqrt(len(KL_numpy))),
          file=open(args.results_file, 'a'))
    del loss
    return KL_mean

@torch.no_grad()
def evaluate_KL(student, teacher, dataloader, args):
    """ evaluate KL of student with recursive single teacher """
    teacher.eval()
    student.eval()

    KL = []

    for i in range(len(dataloader)):

        z_energies = student_energies[(args.batch_size)*(i-1):(args.batch_size)*i]
        z_energy_dist_unit = sample_rec_flow(rec_model, 1, args, z_energies).to('cpu')
        z_energy_dist = trafo_to_energy_space(z_energy_dist_unit, z_energies)

        z_energies = torch.log10(z_energies*3.).unsqueeze(-1)
        z_E0 = torch.log10(z_energy_dist[:, 0].unsqueeze(-1)+1e-8) - 1.
        z_E1 = torch.log10(z_energy_dist[:, 1].unsqueeze(-1)+1e-8) - 1.
        z_E2 = torch.log10(z_energy_dist[:, 2].unsqueeze(-1)+1e-8) - 1.
        z_E3 = torch.log10(z_energy_dist[:, 3].unsqueeze(-1)+1e-8) - 1.
        z_E4 = torch.log10(z_energy_dist[:, 4].unsqueeze(-1)+1e-8) - 1.
    

        z_y = torch.cat((z_energies, z_E0, z_E1, z_E2, z_E3, z_E4), 1).to(args.device)
        noise = (torch.randn(args.batch_size, args.dim_sum)*args.student_width).to(args.device)
        pts, log_student_pre = student._transform.inverse(noise, z_y)
        latent_teacher_noise, log_teacher_pre = teacher._transform.forward(pts, z_y)

        logabsdet_noise_student = logabsdet_of_base(noise, width=args.student_width)
        logabsdet_noise_teacher = logabsdet_of_base(latent_teacher_noise,
                                                    width=args.student_width)


        log_teacher = logabsdet_noise_teacher + log_teacher_pre
        log_student = logabsdet_noise_student - log_student_pre

        KL_local = (log_student-log_teacher.detach()).mean()
        KL.append(log_student-log_teacher.detach())
    KL_mean = torch.cat(KL, dim=0).mean().to('cpu').numpy()
    KL_std = torch.cat(KL, dim=0).std().to('cpu').numpy()
    print("final KL is {} +/- {} (+/- {})".format(KL_mean, KL_std,
                                                  KL_std/ np.sqrt(len(dataloader.dataset))))
    print("final KL is {} +/- {} (+/- {}".format(KL_mean, KL_std,
                                                 KL_std/ np.sqrt(len(dataloader.dataset))),
          file=open(args.results_file, 'a'))

####################################################################################################
#######################################   running the code   #######################################
####################################################################################################

if __name__ == '__main__':
    args = parser.parse_args()
    args.num_layer = 4
    # check if parsed arguments are valid

    assert (args.train or args.evaluate or args.generate_to_file or \
            args.save_only_weights or args.evaluate_KL), (
            "Please specify at least one of --train, --generate, --evaluate, --generate_to_file")

    # check if output_dir exists and 'move' results file there
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    args.results_file = os.path.join(args.output_dir, args.results_file)
    print(args, file=open(args.results_file, 'a'))

    if args.fully_guided:
        args.train_p = True
        args.train_xz = True
    # setup device
    args.device = torch.device('cuda:'+str(args.which_cuda) \
                               if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Using {}".format(args.device))
    print("Using {}".format(args.device), file=open(args.results_file, 'a'))


    # get dataloaders
    if (args.train or args.evaluate or args.evaluate_KL):
        train_dataloader, test_dataloader = get_dataloader('gamma',
                                                        args.data_dir,
                                                        full=False,
                                                        apply_logit=True,
                                                        device=args.device,
                                                        batch_size=args.batch_size,
                                                        with_noise=args.with_noise,
                                                        normed=False,
                                                        normed_layer= True)

    args.input_size = {'0': 8, '1': 160, '2': 190, '3': 5, '4': 5}
    args.input_dims = {'0': (1,8), '1': (10,16), '2': (10,19), '3': (1,5), '4': (1,5)}

    flow_params_rec_energy = {'num_blocks': 2, #num of layers per block
                              'features': args.num_layer+1,
                              'context_features': 1,
                              'hidden_features': 64,
                              'use_residual_blocks': False,
                              'use_batch_norm': False,
                              'dropout_probability': 0.,
                              'activation':getattr(F, args.activation_fn),
                              'random_mask': False,
                              'num_bins': 8,
                              'tails':'linear',
                              'tail_bound': 14,
                              'min_bin_width': 1e-6,
                              'min_bin_height': 1e-6,
                              'min_derivative': 1e-6}
    rec_flow_blocks = []
    for _ in range(6):
        rec_flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_rec_energy))
        rec_flow_blocks.append(transforms.RandomPermutation(args.num_layer+1))
    rec_flow_transform = transforms.CompositeTransform(rec_flow_blocks)
    # _sample not implemented:
    #rec_flow_base_distribution = distributions.DiagonalNormal(shape=[args.num_layer+1])
    rec_flow_base_distribution = distributions.StandardNormal(shape=[args.num_layer+1])
    rec_flow = flows.Flow(transform=rec_flow_transform, distribution=rec_flow_base_distribution)

    rec_model = rec_flow.to(args.device)
    rec_optimizer = torch.optim.Adam(rec_model.parameters(), lr=1e-4)
    print(rec_model)
    print(rec_model, file=open(args.results_file, 'a'))

    total_parameters = sum(p.numel() for p in rec_model.parameters() if p.requires_grad)

    print("Recursive energy setup has {} parameters".format(int(total_parameters)))
    print("Recursive energy setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))

    if os.path.exists(os.path.join('./rec_energy_flow/', 'gamma.pt')):
        print("loading recursive energy flow")
        print("loading recursive energy flow", file=open(args.results_file, 'a'))
        load_rec_flow(rec_model, args)
    else:
        train_rec_flow(rec_model, train_dataloader, test_dataloader, rec_optimizer, args)
        print("loading recursive energy flow")
        print("loading recursive energy flow", file=open(args.results_file, 'a'))
        load_rec_flow(rec_model, args)

    # to plot losses:
    args.train_loss = []
    args.test_loss = []
    # to keep track of dimensionality in constructing the flows
    args.dim_sum = 0
    args.dim_split = []

    flow_params_RQS = {'num_blocks':args.n_hidden, # num of hidden layers per block
                       'use_residual_blocks':args.use_residual,
                       'use_batch_norm':args.batch_norm,
                       'dropout_probability':args.dropout_probability,
                       'activation':getattr(F, args.activation_fn),
                       'random_mask':False,
                       'num_bins':args.n_bins,
                       'tails':'linear',
                       'tail_bound':args.tail_bound,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}

    # setup flow
    # setup single-type flows (plain or recursive)
    flow_blocks = []
    for layer_id in range(args.num_layer+1):
        current_dim = args.input_size[str(layer_id)]
        args.dim_split.append(current_dim)
    for entry in args.dim_split:
        args.dim_sum += entry
    cond_label_size = (2+args.num_layer)

    for i in range(args.n_blocks):
        flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_RQS,
                features=args.dim_sum,
                context_features=cond_label_size,
                hidden_features=args.hidden_size
            ))
        if i%2 == 0:
            flow_blocks.append(InversionLayer(args.dim_split))
        else:
            flow_blocks.append(RandomPermutationLayer(args.dim_split))

    del flow_blocks[-1]
    if args.student_mode:
        flow_transform = GuidedCompositeTransform(flow_blocks)
    else:
        flow_transform = transforms.CompositeTransform(flow_blocks)
    
    flow_base_distribution = distributions.StandardNormal(shape=[args.dim_sum])
    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(model)
    print(model, file=open(args.results_file, 'a'))

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total setup has {} parameters".format(int(total_parameters)))
    print("Total setup has {} parameters".format(int(total_parameters)),
          file=open(args.results_file, 'a'))


    if not args.student_mode:
        print("Running in teacher mode.")
        if args.train:
            print("Training flow-II teacher...")
            starting_time = time.time()
            train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, args,
                               rec_model=rec_model)
            ending_time = time.time()
            total_time = ending_time-starting_time
            time_string = "Needed {:d} min and {:.1f} s to train flow-II teacher."
            print(time_string.format(int(total_time//60), total_time%60))
            print(time_string.format(int(total_time//60), total_time%60),file=open(args.results_file, 'a'))

        if args.evaluate:
            print("Evaluating flow-II teacher...")
            load_weights(model, args)
            evaluate(model, test_dataloader, args.n_epochs, args)

        if args.generate_to_file:
            print("Generating from teacher to file...")
            load_weights(model, args)
            num_events = 121000

            list1 = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
            list2 = [1048576, 2097152, 4194304]
            energy_list = []
            for energy in list1:
                energy_list = np.concatenate((energy_list,(np.ones(int(10000))*energy/1e5)))
            count = 4
            energy_list = np.concatenate((energy_list,(np.ones(int(5000))*524288/1e5)))
            for energy in list2:
                count -= 1
                energy_list= np.concatenate((energy_list,(np.ones(int(1000*count))*energy/1e5)))
            energy_list = np.reshape(energy_list, (1,121000))
            my_energies = torch.from_numpy(energy_list)
            my_energies = torch.reshape(my_energies,(-1,))
            generate_to_file(model, args, num_events=num_events, energies=my_energies,
                             rec_model=rec_model)

        if args.save_only_weights:
            load_all(model, optimizer, args)
            save_weights(model, args)
    else:
        print("Running in student mode")
        if args.train or args.evaluate_KL:
            print("loading teacher")
            teacher = model
            del model
            load_weights(teacher, args)
            print("done")

        # to plot losses:
        args.train_loss = []
        args.test_loss = []

        # setup student for training
        if args.train_xz or args.train_p:
            teacher_perm = []
        else:
            teacher_perm = torch.arange(0, args.dim_sum)

        if args.train:
            # properly treat teacher permutations when training from scratch
            for elem in teacher._transform._transforms:
                if hasattr(elem, '_permutation'):
                    if args.train_xz or args.train_p:
                        teacher_perm.append(elem._permutation.to('cpu'))
                    else:
                        teacher_perm = torch.index_select(teacher_perm, 0, elem._permutation.to('cpu'))
        else:
            # fill with dummies that are then overwritten in loading.
            if args.train_xz or args.train_p:
                for _ in range(args.n_blocks-1 if (args.train_xz or args.train_p) \
                               else args.student_n_blocks-1):
                    teacher_perm.append(transforms.Permutation(torch.arange(0, args.dim_sum)))
            else:
                teacher_perm = transforms.Permutation(torch.arange(0, args.dim_sum))
            
        if args.train_xz or args.train_p:
            teacher_perm.append(teacher_perm[-1])
        flow_blocks = []
        student_perms = []
        for i in range(args.n_blocks if (args.train_xz or args.train_p) \
                       else args.student_n_blocks):
            flow_blocks.append(
                transforms.InverseTransform(
                    #transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    IAFRQS(
                        **flow_params_RQS,
                        features=args.dim_sum,
                        context_features=cond_label_size,
                        hidden_features=args.student_hidden_size
                    )))

            if i%2 == 0:
                flow_blocks.append(InversionLayer(args.dim_split))
            else:
                if args.train:
                    if args.train_xz or args.train_p:
                        flow_blocks.append(transforms.Permutation(teacher_perm[i]))
                    else:
                        flow_blocks.append(RandomPermutationLayer(args.dim_split))
                else: 
                    # add dummy permutation that will be overwritten with loaded model
                    flow_blocks.append(transforms.Permutation(torch.arange(0, args.dim_sum)))

            student_perms.append(flow_blocks[-1]._permutation)
        del flow_blocks[-1]
        del student_perms[-1]
        if not (args.train_xz or args.train_p):
            if not args.train:
                # overwrite teacher_perm, so teacher model is not needed, unless in training
                teacher_perm = torch.arange(0, args.dim_sum)

            student_perms.reverse()
            final_perm = torch.arange(0, args.dim_sum)
            for perm in student_perms:
                final_perm = torch.index_select(final_perm, 0, torch.argsort(perm))
            final_perm = torch.index_select(final_perm, 0, teacher_perm)
            flow_blocks.append(transforms.Permutation(final_perm))
            flow_transform = transforms.CompositeTransform(flow_blocks)

        else:
            flow_transform = GuidedCompositeTransform(flow_blocks)
        
        flow_base_distribution = distributions.StandardNormal(shape=[args.dim_sum])
        student = flows.Flow(transform=flow_transform,
                             distribution=flow_base_distribution).to(args.device)

        optimizer_student = torch.optim.Adam(student.parameters(), lr=args.lr)
        print(student)
        print(student, file=open(args.results_file, 'a'))

        total_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print("Student has {} parameters".format(int(total_parameters)))
        print("Student has {} parameters".format(int(total_parameters)),
              file=open(args.results_file, 'a'))

        if args.train:
            print("Training flow-II for student...")
            starting_time = time.time()
            train_and_evaluate_student(teacher, student, train_dataloader, test_dataloader,
                                       optimizer_student, args, rec_model=rec_model)
            ending_time = time.time()
            total_time = ending_time - starting_time
            time_string = "Needed {:d} min and {:.1f} s to train flow-II for gamma student"
            print(time_string.format(int(total_time//60), total_time%60))

        if args.evaluate_KL:
            print("Evaluating student KL...")
            load_weights(student, args, is_student=True)
            ######## initialize energies for student ############
            list1 = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
            list2 = [1048576, 2097152, 4194304]
            energy_list = []
            for energy in list1:
                energy_list = np.concatenate((energy_list,(np.ones(int(10000))*energy/1e5)))
            count = 4
            energy_list = np.concatenate((energy_list,(np.ones(int(5000))*524288/1e5)))
            for energy in list2:
                count -= 1
                energy_list= np.concatenate((energy_list,(np.ones(int(1000*count))*energy/1e5)))
            energy_list = np.reshape(energy_list, (1,121000))
            student_energies = torch.from_numpy(energy_list)
            student_energies = torch.reshape(student_energies,(-1,))
            student_energies_idx = torch.randperm(student_energies.shape[0])
            student_energies = student_energies[student_energies_idx].view(student_energies.size())
            evaluate_KL(student, teacher, test_dataloader, student_energies, args)
        if args.evaluate:
            print("Evaluating student...")
            load_weights(student, args, is_student=True)
            evaluate(student, test_dataloader, args.n_epochs, args)

        if args.generate_to_file:
            print("Generating from student to file...")
            load_weights(student, args, is_student=True)
            num_events = 121000
            
            list1 = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
            list2 = [1048576, 2097152, 4194304]
            energy_list = []
            for energy in list1:
                energy_list = np.concatenate((energy_list,(np.ones(int(10000))*energy/1e5)))
            count = 4
            energy_list = np.concatenate((energy_list,(np.ones(int(5000))*524288/1e5)))
            for energy in list2:
                count -= 1
                energy_list= np.concatenate((energy_list,(np.ones(int(1000*count))*energy/1e5)))
            energy_list = np.reshape(energy_list, (1,121000))
            my_energies = torch.from_numpy(energy_list)
            my_energies = torch.reshape(my_energies,(-1,))
            print(my_energies.shape)
            generate_to_file(student, args, num_events=num_events, energies=my_energies, #my_energies, #None
                             rec_model=rec_model)

        if args.save_only_weights:
            print("Saving only student weights ...")
            load_all(student, optimizer, args, is_student=True)
            save_weights(student, args, is_student=True)
