import numpy as np
import matplotlib.pyplot as plt
from model import Generator
import torch

latent_dim = 100
nz = 100
ngf = 256
ndf = 256
nc = 1
G = Generator(nc, nz, ngf)


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
    scaled_image_size = 2 * scaled_image_size

    padding = 10
    canvas_size = grid_size * (scaled_image_size + padding) - padding
    canvas = np.zeros((canvas_size, canvas_size))
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            image = generated_fake_images[count]
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