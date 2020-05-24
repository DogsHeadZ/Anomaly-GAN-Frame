"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from models.networks import NetG, NetD, weights_init
from visualizer import Visualizer
from loss import l2_loss
from evaluate import evaluate
from models.basemodel import BaseModel

##
class Ganomaly2(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly2'

    def __init__(self, opt, data):
        super(Ganomaly2, self).__init__(opt, data)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        """
        self.err_g_adv = self.opt.w_adv * self.l_adv(self.feat_fake, self.feat_real)
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_enc(self.latent_o, self.latent_i)
        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()
    
    ##
    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.
        """
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        # if self.err_d.item() < 1e-5: self.reinit_d()

    ##
    def test(self):
        """ Test GANomaly model.

        Args:
            data ([type]): data for the test set

        Raises:
            IOError: Model weights not found.
        """
        self.netg.eval()
        self.netd.eval()
        with torch.no_grad():
            print(f">> Evaluating {self.name} on {self.opt.dataset} to detect {self.opt.abnormal_class}")
            with torch.no_grad():
                # Create big error tensor for the test set.
                an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32,
                                        device=self.device)
                gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long,
                                        device=self.device)

                times = []
                for i, data in enumerate(self.data.valid, 0):
                    time_i = time.time()
                    inputdata = data[0].to(self.device)
                    label = data[1].to(self.device)
                    fake, latent_i, latent_o = self.netg(inputdata)

                    _, feat_real = self.netd(inputdata)
                    _, feat_fake = self.netd(fake)
                    error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)
                    time_o = time.time()

                    an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                        error.size(0))
                    gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = label.reshape(
                        error.size(0))
                    times.append(time_o - time_i)

                times = np.array(times)
                times = np.mean(times[:100] * 1000)

                # Scale error vector between [0, 1]
                an_scores = (an_scores - torch.min(an_scores)) / (
                        torch.max(an_scores) - torch.min(an_scores))
                auc = evaluate(gt_labels, an_scores, metric=self.opt.metric)
                performance = OrderedDict([('Avg Run Time (ms/batch)', times), ('AUC', auc)])
                print(performance)
                return performance
