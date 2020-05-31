from models.networks import *
from evaluate import evaluate
import visualize
import losses
from losses import l2_loss
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm
import time

class Fanogan():

    @property
    def name(self):
        return 'Fanogan'

    def __init__(self, opt, dataloader):
        self.opt = opt
        self.dataloader = dataloader
        self.device = opt.device
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.encoder = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers).to(self.device)
        self.decoder.apply(weights_init)
        self.netd.apply(weights_init)
        self.encoder.apply(weights_init)

    def train(self):
        opt = self.opt
        real_label = torch.ones(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        fake_label = torch.zeros(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        optimizer_de = optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer_en = optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        l_bce = nn.BCELoss()

        if opt.load_best_weights or opt.load_final_weights:
            self.load_GAN_weights()

        print(f">> Training {self.name} on {opt.dataset} to detect {opt.abnormal_class}")
        loss_de = []
        loss_d = []
        # train GAN
        for epoch in range(opt.nepoch):
            iternum = 0
            lossd = 0
            lossde = 0
            self.decoder.train()
            self.netd.train()
            for i, data in enumerate(
                    tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train), ncols=80)):
                iternum = iternum + 1
                inputdata = data[0].to(self.device)
                z = torch.randn(inputdata.shape[0], opt.nz).to(self.device)
                z = z.unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(z)

                # update netd
                optimizer_d.zero_grad()
                pred_real, feat_real = self.netd(inputdata)
                pred_fake, feat_fake = self.netd(fake_image)
                err_d_real = l_bce(pred_real, real_label)
                err_d_fake = l_bce(pred_fake, fake_label)
                err_d = (err_d_real + err_d_fake) * 0.5
                err_d.backward()
                optimizer_d.step()

                # update netde
                optimizer_de.zero_grad()
                z = torch.randn(inputdata.shape[0], opt.nz).to(self.device)
                z = z.unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(z)
                pred_fake, feat_fake = self.netd(fake_image)
                err_g = l_bce(pred_fake, real_label)
                err_g.backward()
                optimizer_de.step()

                # record loss
                lossd += err_d.item()
                lossde += err_g.item()
                # record loss
                if iternum % opt.loss_iter == 0:
                    # print('GLoss: {:.8f} DLoss: {:.8f}'
                    #       .format(running_loss_g / opt.loss_iter, running_loss_d / opt.loss_iter))
                    loss_d.append(lossd / opt.loss_iter)
                    loss_de.append(lossde / opt.loss_iter)
                    lossd = 0
                    lossde = 0

                if opt.save_train_images and i == 0:
                    train_img_dst = os.path.join(opt.outtrain_dir, 'images')
                    visualize.save_images(train_img_dst, epoch, inputdata, fake_image)
            print(">> Training model %s. Epoch %d/%d" % (self.name, epoch + 1, opt.nepoch))

        if opt.save_final_weight:
            self.save_GAN_weights(opt.nepoch)

        if opt.save_loss_curve:
            visualize.plot_loss_curve(opt.outclass_dir, 'loss_d', loss_d)
            visualize.plot_loss_curve(opt.outclass_dir, 'loss_de', loss_de)
        # train encoder
        print(f">> Training {self.name} on {opt.dataset} to detect {opt.abnormal_class}")
        if opt.load_best_en_weights or opt.load_final_en_weights:
            self.load_encoder_weights()
        best_auc = 0
        loss_en = []
        self.netd.eval()
        self.decoder.eval()
        for epoch in range(opt.nenepoch):
            iternum = 0
            lossen = 0
            self.encoder.train()
            for i, data in enumerate(
                    tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train), ncols=80)):
                iternum = iternum + 1
                inputdata = data[0].to(self.device)
                # update encoder
                optimizer_en.zero_grad()
                latent_z = self.encoder(inputdata).squeeze()
                latent_z = latent_z.unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(latent_z)
                pred_real, feat_real = self.netd(inputdata)
                pred_fake, feat_fake = self.netd(fake_image)
                k = 1.0
                error_en = 1 / (inputdata.view(inputdata.shape[0], -1).shape[1]) * l2_loss(inputdata, fake_image) \
                           + k * 1 / (feat_real.view(feat_real.shape[0], -1).shape[1]) * l2_loss(feat_real, feat_fake)
                error_en.backward()
                optimizer_en.step()

                # record loss
                lossen += error_en.item()
                # record loss
                if iternum % opt.loss_iter == 0:
                    # print('GLoss: {:.8f} DLoss: {:.8f}'
                    #       .format(running_loss_g / opt.loss_iter, running_loss_d / opt.loss_iter))
                    loss_en.append(lossen / opt.loss_iter)
                    lossen = 0

                if opt.save_train_images and i == 0:
                    train_img_dst = os.path.join(opt.outtrain_dir, 'images2')
                    visualize.save_images(train_img_dst, epoch, inputdata, fake_image)
            print(">> Training model %s. Epoch %d/%d" % (self.name, epoch + 1, opt.nenepoch))

            if epoch % opt.eva_epoch == 0:
                performance = self.evaluate(epoch)
                if performance['AUC'] > best_auc:
                    best_auc = performance['AUC']
                    if opt.save_best_en_weight:
                        self.save_encoder_weights(epoch, is_best=True)
                # log performance
                now = time.strftime("%c")
                traintime = f'================ {now} ================\n'
                visualize.write_to_log_file(os.path.join(opt.outclass_dir, 'auc.log'), traintime)
                visualize.log_current_performance(opt.outclass_dir, performance, best_auc)
                print(performance)

        if opt.save_final_en_weight:
            self.save_encoder_weights(opt.nepoch)

        if opt.save_loss_curve:
            visualize.plot_loss_curve(opt.outclass_dir, 'loss_en', loss_en)
        print(">> Training model %s.[Done]" % self.name)

    def evaluate(self, epoch):
        self.netd.eval()
        self.decoder.eval()
        self.encoder.eval()
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
                latent_z = self.encoder(inputdata).squeeze()
                latent_z = latent_z.unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(latent_z)
                pred_real, feat_real = self.netd(inputdata)
                pred_fake, feat_fake = self.netd(fake_image)
                k = 1.0
                rec_image = torch.pow((inputdata - fake_image), 2)
                rec_image = rec_image.view(rec_image.shape[0], -1)
                rec_feature = torch.pow((feat_real - feat_fake), 2)
                rec_feature = rec_feature.view(rec_feature.shape[0], -1)
                error = 1 / (rec_image.shape[1]) * torch.mean(rec_image, dim=1) + k * 1 / (rec_feature.shape[1]) * torch.mean(rec_feature, dim=1)
                time_o = time.time()

                an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = label.reshape(
                    error.size(0))

                if self.opt.save_test_images and i == 0:
                    test_img_dst = os.path.join(self.opt.outfolder, self.opt.name, self.opt.abnormal_class, 'test',
                                                'images')
                    visualize.save_images(test_img_dst, epoch, inputdata, fake_image)

                if self.opt.visulize_feature and i == 0:
                    feature_img_dst = os.path.join(self.opt.outtrain_dir, 'features')
                    visualize.tsne_3D(feature_img_dst, epoch, 'feature',
                                      feat_real.reshape(feat_real.size(0), -1).cpu().numpy(),
                                      label.reshape(label.size(0), -1).cpu().numpy())
                    visualize.tsne_2D(feature_img_dst, epoch, 'feature',
                                      feat_real.reshape(feat_real.size(0), -1).cpu().numpy(),
                                      label.reshape(label.size(0), -1).cpu().numpy())

                times.append(time_o - time_i)

            times = np.array(times)
            times = np.mean(times[:100] * 1000)

            # Scale error vector between [0, 1]
            an_scores = (an_scores - torch.min(an_scores)) / (
                    torch.max(an_scores) - torch.min(an_scores))
            auc = evaluate(gt_labels, an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', times), ('AUC', auc)])
            return performance

    def save_GAN_weights(self, epoch: int):
        """Save netG and netD weights for the current epoch.
        Args:
            :param epoch: current epoch of model weights
            :param is_best: whether the best model weights
        """
        weight_dir = os.path.join(self.opt.outtrain_dir, 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_final.pth")
        torch.save({'epoch': epoch, 'state_dict': self.decoder.state_dict()}, f"{weight_dir}/decoder_final.pth")

    def load_GAN_weights(self):
        weight_dir = os.path.join(self.opt.outtrain_dir, 'weights')
        if self.opt.load_final_weights:
            weight_g_path = f'{weight_dir}/decoder_final.pth'
            weight_d_path = f'{weight_dir}/netD_final.pth'
        print('>> Loading GAN weights...')
        weights_g = torch.load(weight_g_path)['state_dict']
        weights_d = torch.load(weight_d_path)['state_dict']
        try:
            self.decoder.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("GAN weights not found")
        print('   Done.')

    def save_encoder_weights(self, epoch: int, is_best: bool = False):
        """Save netG and netD weights for the current epoch.
        Args:
            :param epoch: current epoch of model weights
            :param is_best: whether the best model weights
        """
        weight_dir = os.path.join(self.opt.outtrain_dir, 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.encoder.state_dict()}, f'{weight_dir}/encoder_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.encoder.state_dict()}, f"{weight_dir}/encoder_final.pth")

    def load_encoder_weights(self):
        weight_dir = os.path.join(self.opt.outtrain_dir, 'weights')
        if self.opt.load_best_en_weights:
            weight_en_path = f'{weight_dir}/encoder_best.pth'
        if self.opt.load_final_en_weights:
            weight_en_path = f'{weight_dir}/encoder_final.pth'
        print('>> Loading Encoder weights...')
        weights_en = torch.load(weight_en_path)['state_dict']
        try:
            self.encoder.load_state_dict(weights_en)
        except IOError:
            raise IOError("encoder weights not found")
        print('   Done.')