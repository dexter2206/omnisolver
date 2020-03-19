import dimod
import torch
import time
import numpy as np
from functools import partial
import random
from ...van_orig.src_hop_sk.sk import SKModel
from ...van_orig.src_hop_sk.made2 import MADE

class VanSampler(dimod.Sampler):
    """Implementatoin of simple random-sampler."""

    def __init__(self, **kwargs):
        d = kwargs.pop('cuda')
        self.n = kwargs.pop('n')
        self.device = 'cuda:{}'.format(d) if d >=0 else 'cpu'
        self.net_params = {
            'n': self.n,
            'net_depth': kwargs.pop('net_depth'),
            'net_width': kwargs.pop('net_width'),
            'bias': kwargs.pop('bias', False),
            'z2': kwargs.pop('z2', False),
            'res_block': kwargs.pop('res_block', False),
            'x_hat_clip': kwargs.pop('x_hat_clip'),
            'epsilon': kwargs.pop('epsilon'),
            'device': self.device
        }


    def _prepare_sampling(self, bqm, **kwargs):
        self.ham = ModifiedSKModel(self.n, self.device, bqm)
        self.ham.J.requires_grad = False


        self.net = MADE(**self.net_kwargs)
        self.net.to(self.device)

    def my_log(self, s):
        if self.out_filename:
            with open(self.out_filename + '.log', 'a', newline='\n') as f:
                f.write(s + u'\n')
        if not self.no_stdout:
            print(s)

    def sample(self, bqm, **kwargs):
        start_time = time.time()
        self._prepare_sampling(bqm)
        self.__dict__.update(kwargs)

        params = list(self.net.parameters())
        params = list(filter(lambda p: p.requires_grad, params))
        nparams = int(sum([np.prod(p.shape) for p in params]))
        self.my_log('Total number of trainable parameters: {}'.format(nparams))

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.lr)
        elif self.optimizer == 'sgdm':
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=self.lr, alpha=0.99)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        elif self.optimizer == 'adam0.5':
            optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.999))
        else:
            raise ValueError('Unknown optimizer: {}'.format(self.optimizer))

        init_time = time.time() - start_time
        self.my_log('init_time = {:.3f}'.format(init_time))

        self.my_log('Training...')
        sample_time = 0
        train_time = 0
        start_time = time.time()
        if self.beta_anneal_to < self.beta:
            self.beta_anneal_to = self.beta
        beta = self.beta
        while beta <= self.beta_anneal_to:
            for step in range(self.max_step):
                optimizer.zero_grad()

                sample_start_time = time.time()
                with torch.no_grad():
                    sample, x_hat = net.sample(self.batch_size)
                assert not sample.requires_grad
                assert not x_hat.requires_grad
                sample_time += time.time() - sample_start_time

                train_start_time = time.time()

                log_prob = self.net.log_prob(sample)
                with torch.no_grad():
                    energy = self.ham.energy(sample)
                    loss = log_prob + beta * energy
                assert not energy.requires_grad
                assert not loss.requires_grad
                loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
                loss_reinforce.backward()

                if self.clip_grad > 0:
                    # nn.utils.clip_grad_norm_(params, self.clip_grad)
                    parameters = list(filter(lambda p: p.grad is not None, params))
                    max_norm = float(self.clip_grad)
                    norm_type = 2
                    total_norm = 0
                    for p in parameters:
                        param_norm = p.grad.data.norm(norm_type)
                        total_norm += param_norm.item()**norm_type
                        total_norm = total_norm**(1 / norm_type)
                        clip_coef = max_norm / (total_norm + self.epsilon)
                        for p in parameters:
                            p.grad.data.mul_(clip_coef)

                optimizer.step()

                train_time += time.time() - train_start_time

                if self.print_step and step % self.print_step == 0:
                    free_energy_mean = loss.mean() / beta / self.n
                    free_energy_std = loss.std() / beta / self.n
                    entropy_mean = -log_prob.mean() / self.n
                    energy_mean = energy.mean() / self.n
                    mag = sample.mean(dim=0)
                    mag_mean = mag.mean()
                    if step > 0:
                        sample_time /= self.print_step
                        train_time /= self.print_step
                    used_time = time.time() - start_time
                    self.my_log(
                        'beta = {:.3g}, # {}, F = {:.8g}, F_std = {:.8g}, S = {:.5g}, E = {:.5g}, M = {:.5g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                        .format(
                            beta,
                            step,
                            free_energy_mean.item(),
                            free_energy_std.item(),
                            entropy_mean.item(),
                            energy_mean.item(),
                            mag_mean.item(),
                            sample_time,
                            train_time,
                            used_time,
                        ))
                    sample_time = 0
                    train_time = 0

            with open(self.fname, 'a', newline='\n') as f:
                f.write('{} {} {:.3g} {:.8g} {:.8g} {:.8g} {:.8g}\n'.format(
                    self.n,
                    self.seed,
                    beta,
                    free_energy_mean.item(),
                    free_energy_std.item(),
                    energy_mean.item(),
                    entropy_mean.item(),
                ))

            beta += self.beta_inc

        return dimod.SampleSet.from_samples(
            sample, energy=energy, vartype=bqm.vartype
        )

    @property
    def properties(self):
        return dict()

    @property
    def parameters(self):
        return {"num_reads": []}


class ModifiedSKModel(SKModel):
    def __init__(self, n, device, bqm, field=0, seed=0):
        self.n = n
        self.field = field
        self.seed = seed
        if seed > 0:
            torch.manual_seed(seed)
        J = bqm.to_numpy_matrix()
        self.J = torch.tensor(J, dtype=torch.float32)
        # Symmetric matrix, zero diagonal
        self.J = self.J.to(device)
        self.J.requires_grad = True

        self.C_model = []

        print('SK model with n = {},  field = {}, seed = {}'.format(
            n, field, seed))