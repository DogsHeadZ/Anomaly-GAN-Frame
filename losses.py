"""
Losses
"""
# pylint: disable=C0301,C0103,R0902,R0915,W0221,W0622


##
# LIBRARIES
import torch

import torch
import torch.nn.functional as F


__all__ = ['kl', 'reconstruction', 'discriminator_logistic_simple_gp',
           'discriminator_gradient_penalty', 'generator_logistic_non_saturating']


def kl(mu, log_var):
    return -0.5 * torch.mean(torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), 1))


def reconstruction(recon_x, x, lod=None):
    return torch.mean((recon_x - x)**2)


def discriminator_logistic_simple_gp(d_result_fake, d_result_real, reals, r1_gamma=10.0):
    loss = (F.softplus(d_result_fake) + F.softplus(-d_result_real))

    if r1_gamma != 0.0:
        real_loss = d_result_real.sum()
        real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
        r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
        loss = loss + r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def discriminator_gradient_penalty(d_result_real, reals, r1_gamma=10.0):
    real_loss = d_result_real.sum()
    real_grads = torch.autograd.grad(real_loss, reals, create_graph=True, retain_graph=True)[0]
    r1_penalty = torch.sum(real_grads.pow(2.0), dim=[1, 2, 3])
    loss = r1_penalty * (r1_gamma * 0.5)
    return loss.mean()


def generator_logistic_non_saturating(d_result_fake):
    return F.softplus(-d_result_fake).mean()


##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)
