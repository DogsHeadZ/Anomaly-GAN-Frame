import os
import time
import numpy as np
import torchvision.utils as vutils


def save_images(path, epoch, real_images, fake_images):

    vutils.save_image(real_images, '%s/real_images_%03d.png' % (path, epoch + 1), normalize=True)
    vutils.save_image(fake_images, '%s/fake_images_%03d.png' % (path, epoch + 1), normalize=True)