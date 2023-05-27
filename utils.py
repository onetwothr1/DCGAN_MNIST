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

def generate_images(epoch, path, fixed_noise, num_test_samples, noise_size, netG, device, use_fixed=False):
    '''
    reference: https://github.com/AKASHKADEL/dcgan-mnist
    '''
    z = torch.randn(num_test_samples, noise_size, 1, 1, device=device)
    grid_size = int(math.sqrt(num_test_samples))
  
    if use_fixed:
        generated_fake_images = netG(fixed_noise)
        path += 'fixed_noise/'
    else:
        generated_fake_images = netG(z)
        path += 'variable_noise/'
    label = 'Epoch_{}'.format(epoch)

    image_size = 28
    scaled_image_size = 2 * image_size
    padding = 10
    canvas_size = grid_size * (scaled_image_size + padding) - padding
    canvas = np.zeros((canvas_size, canvas_size))

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            image = generated_fake_images[count].squeeze().detach().numpy()
            print(image.size())
            # print(type(image))
            # print(image[0])
            image_resized = np.kron(image, np.ones(image_size))
            start_row = i * (image_size + padding)
            end_row = start_row + image_size
            start_col = j * (image_size + padding)
            end_col = start_col + image_size
            canvas[start_row:end_row, start_col:end_col] = image_resized
            count += 1

    plt.imshow(canvas, cmap='gray')
    plt.axis('off')
    plt.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+label+'.png')
    plt.close()

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