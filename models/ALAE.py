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


def loss_ed(pred_real, pred_fake):
    loss = torch.log(1+torch.exp(pred_fake))+torch.log(1+torch.exp(-pred_real))
    return torch.mean(loss)


def loss_fg(pred_fake):
    loss = torch.log(1+torch.exp(-pred_fake))
    return torch.mean(loss)


def loss_eg(latent_i, latent_o):
    loss = torch.mean(torch.pow((latent_i - latent_o), 2))
    return loss


class Alae():

    @property
    def name(self):
        return 'Alae'

    def __init__(self, opt, dataloader):
        self.opt = opt
        self.dataloader = dataloader
        self.device = opt.device
        self.netf = NetF(self.opt).to(self.device)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers).to(self.device)
        self.encoder = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers).to(self.device)
        self.netd = NetDFC(self.opt).to(self.device)
        self.decoder.apply(weights_init)
        self.encoder.apply(weights_init)

    def train(self):
        opt = self.opt
        real_label = torch.ones(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        fake_label = torch.zeros(size=(opt.batchsize,), dtype=torch.float32, device=self.device)
        optimizer_f = optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer_de = optim.Adam(self.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer_en = optim.Adam(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer_d = optim.Adam(self.netd.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        l_bce = nn.BCELoss()
        l_adv = l2_loss
        l_con = nn.L1Loss()
        l_enc = l2_loss

        if opt.load_best_weights or opt.load_final_weights:
            self.load_weights()

        print(f">> Training {self.name} on {opt.dataset} to detect {opt.abnormal_class}")
        best_auc = 0
        loss_eds = []
        loss_fgs = []
        loss_egs = []
        for epoch in range(opt.nepoch):
            iternum = 0
            lossed = 0.0
            lossfg = 0.0
            losseg = 0.0
            self.netf.train()
            self.decoder.train()
            self.encoder.train()
            self.netd.train()
            for i, data in enumerate(tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train), ncols=80)):
                iternum = iternum + 1
                data[0].requires_grad = True
                inputdata = data[0].to(self.device)
                z = torch.randn(inputdata.shape[0], opt.nz).to(self.device)
                latent_i = self.netf(z).unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(latent_i)
                latent_o = self.encoder(fake_image).squeeze()
                real_latent = self.encoder(inputdata).squeeze()
                pred_real = self.netd(real_latent)
                pred_fake = self.netd(latent_o)
                optimizer_en.zero_grad()
                optimizer_d.zero_grad()
                loss = loss_ed(pred_real, pred_fake)
                # loss = losses.discriminator_logistic_simple_gp(pred_fake, pred_real, inputdata)
                lossed += loss.item()
                loss.backward()
                optimizer_en.step()
                optimizer_d.step()

                z = torch.randn(inputdata.shape[0], opt.nz).to(self.device)
                latent_i = self.netf(z).unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(latent_i)
                latent_o = self.encoder(fake_image).squeeze()
                pred_fake = self.netd(latent_o)
                optimizer_f.zero_grad()
                optimizer_de.zero_grad()
                loss = loss_fg(pred_fake)
                # loss = losses.generator_logistic_non_saturating(pred_fake)
                lossfg += loss.item()
                loss.backward()
                optimizer_f.step()
                optimizer_de.step()

                z = torch.randn(inputdata.shape[0], opt.nz).to(self.device)
                latent_i = self.netf(z).unsqueeze(2).unsqueeze(3)
                fake_image = self.decoder(latent_i)
                latent_o = self.encoder(fake_image).squeeze()
                optimizer_en.zero_grad()
                optimizer_de.zero_grad()
                loss = l2_loss(latent_i, latent_o)
                # loss = losses.reconstruction(latent_i, latent_o)
                losseg += loss.item()
                loss.backward()
                optimizer_en.step()
                optimizer_de.step()

                # record loss
                if iternum % opt.loss_iter == 0:
                    # print('GLoss: {:.8f} DLoss: {:.8f}'
                    #       .format(running_loss_g / opt.loss_iter, running_loss_d / opt.loss_iter))
                    loss_eds.append(lossed / opt.loss_iter)
                    loss_fgs.append(lossfg / opt.loss_iter)
                    loss_egs.append(losseg / opt.loss_iter)
                    lossed = 0
                    lossfg = 0
                    losseg = 0

                if opt.save_train_images and i == 0:
                    train_img_dst = os.path.join(opt.outtrain_dir, 'images')
                    visualize.save_images(train_img_dst, epoch, inputdata, fake_image)
            print(">> Training model %s. Epoch %d/%d" % (self.name, epoch + 1, opt.nepoch))

            # if epoch % opt.eva_epoch == 0:
            #     performance = self.evaluate(epoch)
            #     if performance['AUC'] > best_auc:
            #         best_auc = performance['AUC']
            #         if opt.save_best_weight:
            #             self.save_weights(epoch, is_best=True)
            #     # log performance
            #     now = time.strftime("%c")
            #     traintime = f'================ {now} ================\n'
            #     visualize.write_to_log_file(os.path.join(opt.outclass_dir, 'auc.log'), traintime)
            #     visualize.log_current_performance(opt.outclass_dir, performance, best_auc)
            #     print(performance)
        if opt.save_loss_curve:
            visualize.plot_loss_curve(opt.outclass_dir, 'loss_ed', loss_eds)
            visualize.plot_loss_curve(opt.outclass_dir, 'loss_fg', loss_fgs)
            visualize.plot_loss_curve(opt.outclass_dir, 'loss_eg', loss_egs)
        # if opt.save_final_weight:
        #     self.save_weights(opt.nepoch, is_best=False)
        print(">> Training model %s.[Done]" % self.name)

    def evaluate(self, epoch):
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

                if self.opt.save_test_images and i == 0:
                    test_img_dst = os.path.join(self.opt.outfolder, self.opt.name, self.opt.abnormal_class, 'test',
                                                'images')
                    visualize.save_images(test_img_dst, epoch, inputdata, fake)

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

    def save_weights(self, epoch: int, is_best: bool = False):
        """Save netG and netD weights for the current epoch.
        Args:
            :param epoch: current epoch of model weights
            :param is_best: whether the best model weights
        """
        weight_dir = os.path.join(self.opt.outtrain_dir, 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/netG_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/netD_final.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/netG_final.pth")

    def load_weights(self):
        weight_dir = os.path.join(self.opt.outtrain_dir, 'weights')
        if self.opt.load_best_weights:
            weight_g_path = f'{weight_dir}/netG_best.pth'
            # weight_g_path = f"{self.opt.outfolder}/{self.name}/{self.opt.dataset}/" \
            #                 f"{self.opt.abnormal_class}/train/weights/netG_best.pth"
            weight_d_path = f'{weight_dir}/netD_best.pth'
            # weight_d_path = f"{self.opt.outfolder}/{self.name}/{self.opt.dataset}/" \
            #                 f"{self.opt.abnormal_class}/train/weights/netD_best.pth"
        if self.opt.load_final_weights:
            weight_g_path = f'{weight_dir}/netG_final.pth'
            # weight_g_path = f"{self.opt.outfolder}/{self.name}/{self.opt.dataset}/" \
            #                 f"{self.opt.abnormal_class}/train/weights/netG_final.pth"
            weight_d_path = f'{weight_dir}/netD_final.pth'
            # weight_d_path = f"{self.opt.outfolder}/{self.name}/{self.opt.dataset}/" \
            #                 f"{self.opt.abnormal_class}/train/weights/netD_final.pth"
        print('>> Loading weights...')
        weights_g = torch.load(weight_g_path)['state_dict']
        weights_d = torch.load(weight_d_path)['state_dict']
        try:
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')