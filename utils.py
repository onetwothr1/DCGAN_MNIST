import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.utils import make_grid
import math
import itertools

def plot(train_loss_d, train_loss_g, p_real, p_fake):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

  ax1.plot(train_loss_d, label='D train loss')
  ax1.plot(train_loss_g, label='G train loss')
  ax1.set_xlabel('epoch')
  ax1.set_xlabel('loss')
  ax1.legend(bbox_to_anchor=(0.8, 1), loc=2, borderaxespad=0.)

  ax2.plot(p_real, label='p_real')
  ax2.plot(p_fake, label='p_fake')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('p')
  ax2.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)

  plt.show()

def imshow(img):
    img = (img+1)/2    
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = make_grid(img.cpu().detach())
    img = (img+1)/2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def generate_images(epoch, path, fixed_noise, num_test_samples, netG, device, use_fixed=False):
    '''
    reference: https://github.com/AKASHKADEL/dcgan-mnist
    '''
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
        title = 'Fixed Noise'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
        title = 'Variable Noise'
  
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6,6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i,j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k//4
        j = k%4
        ax[i,j].cla()
        ax[i,j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28,28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch+1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path+label+'.png')

def save_gif(path, fps, fixed_noise=False):
    '''
    reference: https://github.com/AKASHKADEL/dcgan-mnist
    '''
    if fixed_noise==True:
        path += 'fixed_noise/'
    else:
        path += 'variable_noise/'
    images = glob(path + '*.png')
    images = natsort.natsorted(images)
    gif = []

    for image in images:
        gif.append(imageio.imread(image))
    imageio.mimsave(path+'animated.gif', gif, fps=fps)