from models.networks import NetG, NetD, weights_init
from evaluate import evaluate
from loss import l2_loss
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm
import time


class Myganomaly2():

    @property
    def name(self):
        return 'Myganomaly2'

    def __init__(self, opt, dataloader):
        self.opt = opt
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)

        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)

        self.real_label = torch.ones(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.l_bce = nn.BCELoss()
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss

    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))


    def train(self):
        opt = self.opt
        # device to be finished==================================================


        print(f">> Training {self.name} on {opt.dataset} to detect {opt.abnormal_class}")
        self.netd.train()
        self.netg.train()
        best_auc = 0
        for epoch in range(opt.nepoch):
            iternum = 0
            self.netd.train()
            self.netg.train()
            for data in tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train), ncols=80):
                iternum = iternum + 1
                self.set_input(data)
                # self.optimize_params()

                fake, latent_i, latent_o = self.netg(self.input)

                pred_real, feat_real = self.netd(self.input)
                pred_fake, feat_fake = self.netd(fake.detach())

                self.optimizer_g.zero_grad()
                err_g_adv = self.opt.w_adv * self.l_adv(feat_fake, feat_real)
                err_g_con = self.opt.w_con * self.l_con(fake, self.input)
                err_g_lat = self.opt.w_lat * self.l_enc(latent_o, latent_i)
                err_g = err_g_adv + err_g_con + err_g_lat
                err_g.backward(retain_graph=True)
                self.optimizer_g.step()

                self.optimizer_d.zero_grad()
                err_d_real = self.l_bce(pred_real, self.real_label)
                err_d_fake = self.l_bce(pred_fake, self.fake_label)
                err_d = (err_d_real + err_d_fake) * 0.5
                err_d.backward()
                self.optimizer_d.step()
            print(">> Training model %s. Epoch %d/%d" % (self.name, epoch + 1, opt.niter))

            if epoch % opt.save_epoch == 0:
                res = self.evaluate()
                if res['AUC'] > best_auc:
                    best_auc = res['AUC']
                    if opt.save_best_weight:
                        self.save_weights(epoch, is_best=False)

        if opt.save_final_weight:
            self.save_weights(opt.nepoch, is_best=False)
        print(">> Training model %s.[Done]" % self.name)

    def evaluate(self):
        self.netd.eval()
        self.netg.eval()
        print(f">> Evaluating {self.name} on {self.opt.dataset} to detect {self.opt.abnormal_class}")
        with torch.no_grad():
            # Create big error tensor for the test set.
            an_scores = torch.zeros(size=(len(self.dataloader.valid.dataset),), dtype=torch.float32, device=self.device)
            gt_labels = torch.zeros(size=(len(self.dataloader.valid.dataset),), dtype=torch.long, device=self.device)

            times = []
            for i, data in enumerate(self.dataloader.valid, 0):
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

    def save_weights(self, epoch: int, is_best: bool = False):
        """Save netG and netD weights for the current epoch.

        Args:
            :param epoch: current epoch of model weights
            :param is_best: whether the best model weights
        """
        weight_dir = os.path.join(self.opt.outfolder, self.opt.name, self.opt.abnormal_class, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/netG_best_{epoch}.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/netD_best_{epoch}.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_{epoch}.pth")
