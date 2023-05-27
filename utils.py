import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math

def he_initialization(parameters, activation, negative_slope=0):
    for param in parameters:
        if len(param.size()) > 1:
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity=activation, a=negative_slope)
        else:
            nn.init.zeros_(param)

def save_graph(train_loss_d, train_loss_g, p_real, p_fake, path):
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

    plt.savefig(path)

def image_resize(image, image_size, k):
    resized = np.zeros((image_size * k, image_size * k))
    for i in range(image_size):
        for j in range(image_size):
            resized[i*k : (i+1)*k,  j*k : (j+1)*k] = image[i][j]
    return resized

def generate_images(epoch, path, fixed_noise, num_test_samples, netG):
    generated_fake_images = netG(fixed_noise)
  
    grid_size = int(math.sqrt(num_test_samples))
    image_size = 28
    resize_factor = 3
    scaled_image_size = resize_factor * image_size
    padding = 5
    outer_margin = 30
    canvas_size = grid_size * (scaled_image_size + padding) - padding + outer_margin * 2
    canvas = np.ones((canvas_size, canvas_size))

    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            image = generated_fake_images[count].squeeze().detach().numpy()
            # image = (image * 255).astype(np.uint8) # to integer
            image_resized = image_resize(image, image_size, resize_factor)

            start_row = outer_margin + i * (scaled_image_size + padding)
            end_row = start_row + scaled_image_size
            start_col = outer_margin + j * (scaled_image_size + padding)
            end_col = start_col + scaled_image_size
            canvas[start_row:end_row, start_col:end_col] = image_resized
            count += 1
    canvas = (canvas * 255).astype(np.uint8)

    # draw text (epoch number)
    text_img = text_image(f'Epoch {epoch}', (canvas_size, 100), 30, path)

    # concat image and text
    combined_image = np.concatenate((text_img, canvas), axis=0)
    combined_image_path = path + f'Epoch {epoch}.png'
    plt.imsave(combined_image_path, combined_image, cmap='gray')


def text_image(text, image_size, font_size, path):
    # Create a blank figure with the desired size
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100))

    # Set the axes limits to match the image size
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(0, image_size[1])

    # Disable axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Set the font properties for the text
    font_properties = {'family': 'monospace', 'size': font_size}

    # Render the text as grayscale pixels
    ax.text(image_size[0]/2, image_size[1]/2, text, ha='center', va='center', **font_properties, color='black')

    # Convert the plot to a grayscale numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    gray_data = np.dot(data[..., :3], [0.2989, 0.5870, 0.1140])  # Convert RGB to grayscale

    # Resize the grayscale image to the desired size using PIL
    image = Image.fromarray(gray_data)
    resized_image = image.resize(image_size, resample=Image.LANCZOS)

    # Convert the image mode to grayscale
    resized_image = resized_image.convert("L")

    return np.array(resized_image)

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