import os
import time
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


def save_images(path, epoch, real_images, fake_images):
    if not os.path.isdir(path):
        os.makedirs(path)
    vutils.save_image(real_images, '%s/real_images_%03d.png' % (path, epoch + 1), normalize=True)
    vutils.save_image(fake_images, '%s/fake_images_%03d.png' % (path, epoch + 1), normalize=True)


def tsne_3D(path, epoch, name, feature, label):
    if not os.path.isdir(path):
        os.makedirs(path)
    X_norm = tsne_norm(feature, 3)
    # 画3D特征图
    zeroindex = np.where(label == 0)
    x1 = X_norm[zeroindex, 0]
    y1 = X_norm[zeroindex, 1]
    z1 = X_norm[zeroindex, 2]
    oneindex = np.where(label == 1)
    x2 = X_norm[oneindex, 0]
    y2 = X_norm[oneindex, 1]
    z2 = X_norm[oneindex, 2]
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1, y1, z1, c='r', label='normal')
    ax.scatter(x2, y2, z2, c='g', label='anomaly')
    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.savefig(f'{path}/tsne3D_{name}_{epoch}.png')
    plt.close()


def tsne_2D(path, epoch, name, feature, label):
    if not os.path.isdir(path):
        os.makedirs(path)
    X_norm = tsne_norm(feature, 2)
    zeroindex = np.where(label == 0)
    x1 = X_norm[zeroindex, 0]
    y1 = X_norm[zeroindex, 1]
    oneindex = np.where(label == 1)
    x2 = X_norm[oneindex, 0]
    y2 = X_norm[oneindex, 1]
    # 绘制散点图
    plt.figure()
    plt.scatter(x1, y1, marker='o', color='red', s=40, label='normal')
    #                   记号形状       颜色           点的大小    设置标签
    plt.scatter(x2, y2, marker='o', color='blue', s=40, label='anomaly')
    plt.legend(loc='best')  # 设置 图例所在的位置 使用推荐位置
    plt.savefig(f'{path}/tsne2D_{name}_{epoch}.png')
    plt.close()


def tsne_norm(feature, dim):
    tsne = TSNE(n_components=dim, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(feature)
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    return X_norm


def write_to_log_file(log_name, text):
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % text)


def log_current_performance(path, performance, best):
    """ Print current performance results.

    Args:
        performance ([OrderedDict]): Performance of the model
        best ([int]): Best performance.
    """

    message = '   '
    for key, val in performance.items():
        message += '%s: %.3f ' % (key, val)
    message += 'max AUC: %.3f' % best

    write_to_log_file(os.path.join(path, 'auc.log'), message)


def plot_loss_curve(path, name, loss):
    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(f'{path}/{name}.png')
    plt.close()
