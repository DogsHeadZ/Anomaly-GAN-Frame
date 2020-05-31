""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch
import time
import visualize
# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')        
        self.parser.add_argument('--path', default='', help='path to the folder or image to be predicted.')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=32, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='skipganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--outfolder', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--abnormal_class', default='anomaly_class', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--eva_epoch', type=int, default=1,
                                 help='the epoch of evaluation')
        self.parser.add_argument('--loss_iter', type=int, default=1,
                                 help='the iteration of saving loss')
        self.parser.add_argument('--nepoch', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--nenepoch', type=int, default=15, help='number of epochs to train for encoder')
        self.parser.add_argument('--save_best_weight', action='store_false', help='save the weights of the best model')
        self.parser.add_argument('--save_final_weight', action='store_false', help='save the weights of the final model')
        self.parser.add_argument('--save_best_en_weight', action='store_false', help='save the weights of the best encoder model')
        self.parser.add_argument('--save_final_en_weight', action='store_false',
                                 help='save the weights of the final encoder model')
        self.parser.add_argument('--load_best_weights', action='store_true', help='Load the pretrained best weights')
        self.parser.add_argument('--load_final_weights', action='store_true', help='Load the pretrained final weights')
        self.parser.add_argument('--load_best_en_weights', action='store_true', help='Load the pretrained best encoder weights')
        self.parser.add_argument('--load_final_en_weights', action='store_true', help='Load the pretrained final encoder weights')
        self.parser.add_argument('--save_train_images', action='store_false', help='save train images')
        self.parser.add_argument('--save_test_images', action='store_false', help='Save test images')
        self.parser.add_argument('--visulize_feature', action='store_false', help='visulize features')
        self.parser.add_argument('--save_loss_curve', action='store_false', help='Save loss curve in training')



        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='Weight for adversarial loss. default=1')
        self.parser.add_argument('--w_con', type=float, default=50, help='Weight for reconstruction loss. default=50')
        self.parser.add_argument('--w_lat', type=float, default=1, help='Weight for latent space loss. default=1')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.isTrain = True
        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """
        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain   # train or test
        if not torch.cuda.is_available():
            opt.device = 'cpu'
        # Multi-gpu training remains to be implemented
        opt.device = torch.device("cuda:0" if opt.device != 'cpu' else "cpu")
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # # set gpu ids
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        args = vars(opt)
        now = time.strftime("%c")
        traintime = f'================ {now} ================\n'
        if opt.verbose:
            print('------------ Options -------------')
            print(traintime)
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if opt.name == 'experiment_name':
            opt.name = "%s/%s" % (opt.model, opt.dataset)

        opt.outclass_dir = os.path.join(opt.outfolder, opt.name, opt.abnormal_class)
        opt.outtrain_dir = os.path.join(opt.outfolder, opt.name, opt.abnormal_class, 'train')
        opt.outtest_dir = os.path.join(opt.outfolder, opt.name, opt.abnormal_class, 'test')

        if not os.path.isdir(opt.outtrain_dir):
            os.makedirs(opt.outtrain_dir)
        if not os.path.isdir(opt.outtest_dir):
            os.makedirs(opt.outtest_dir)

        file_name = os.path.join(opt.outclass_dir, 'opt.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ Options -------------\n')
            opt_file.write(traintime)
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return opt
