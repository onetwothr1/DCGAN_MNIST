import torch
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from network import Generator, Discriminator
from utils import *

class Train():
    def __init__(self, generator, discriminator, d_noise, batch_size):
        self.G = generator
        self.D = discriminator
        self.optim_g = optim.Adam(self.G.parameters(), lr = 0.0002, betas=(0.5, 0.999))
        self.optim_d = optim.Adam(self.D.parameters(), lr = 0.0002, betas=(0.5, 0.999))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G.to(self.device)
        self.D.to(self.device)

        self.batch_size = batch_size
        self.dataloader()

        self.d_noise = d_noise
        self.fixed_noise = torch.load('fixed_noise.pt')
        self.k = 1

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.d_noise, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def dataloader(self):
        standardizator = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5),  
                                                        std=(0.5))])

        train_data = dsets.MNIST(root='data/', train=True, transform=standardizator, download=True)
        test_data = dsets.MNIST(root='data/', train=False, transform=standardizator, download=True)

        self.train_data_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(test_data, self.batch_size, shuffle=True)

    def evaluate(self):
        p_real, p_fake = 0., 0.
        for test_img, test_label in self.test_data_loader:
            test_img, test_label = test_img.to(self.device), test_label.to(self.device)
            with torch.autograd.no_grad():
                p_real += (torch.sum(self.D(test_img)).item())
                noise = self.sample_z(len(test_img))
                p_fake_list = self.D(self.G(noise))
                p_fake += (torch.sum(p_fake_list).item())
            
        return p_real / len(self.test_data_loader.dataset), p_fake / len(self.test_data_loader.dataset)

    def run_minibatch_d(self, train_img):
        self.optim_d.zero_grad()

        p_real = self.D(train_img)
        noise = self.sample_z(len(train_img))
        p_fake = self.D(self.G(noise))

        loss_real = -1 * torch.log(p_real)
        loss_fake = -1 * torch.log(1 - p_fake)
        loss_d = (loss_real + loss_fake).mean()

        loss_d.backward()
        self.optim_d.step()

        return loss_d.detach().cpu(), p_real.mean().item(), p_fake.mean().item()

    def run_minibatch_g(self, num):
        self.optim_g.zero_grad()

        noise = self.sample_z(num)
        p_fake = self.D(self.G(noise))

        loss_g = -1 * torch.log(p_fake).mean()
        
        loss_g.backward()
        self.optim_g.step()

        return loss_g.detach().cpu()

    def train(self, num_epoch, start_epoch=0, k=1):
        train_loss_d = []
        train_loss_g = []
        test_p_real = []
        test_p_fake = []

        if start_epoch==0:
            generate_images('Epoch 0', 'images/', self.fixed_noise, 64, self.G)

        # training
        for epoch in range(start_epoch, start_epoch+num_epoch):
            train_d_loss_epoch = []
            train_g_loss_epoch = []
            train_p_real_epoch = []
            train_p_fake_epoch = []
            for _iter, (train_img, train_label) in tqdm(enumerate(self.train_data_loader)):
                train_img, train_label = train_img.to(self.device), train_label.to(self.device)
        
                # Train Discriminator
                if _iter % k == 0:
                    loss_d, p_real, p_fake = self.run_minibatch_d(train_img)
                train_d_loss_epoch.append(loss_d)
                train_p_real_epoch.append(p_real)
                train_p_fake_epoch.append(p_fake)

                # Train Generator
                loss_g = self.run_minibatch_g(len(train_img))
                train_g_loss_epoch.append(loss_g)
            
            train_loss_d.append(sum(train_d_loss_epoch) / len(train_d_loss_epoch))
            train_loss_g.append(sum(train_g_loss_epoch) / len(train_g_loss_epoch))

            # show graph per each epoch
            plt.plot(train_d_loss_epoch, label='D_loss')
            plt.plot(train_g_loss_epoch, label='G_loss')
            plt.plot(train_p_real_epoch, label='p_real')
            plt.plot(train_p_fake_epoch, label='p_fake')
            plt.ylim(bottom=0)
            plt.xlabel('batch iteration')
            plt.title(f'Epoch {epoch+1}')
            plt.legend()
            plt.show()
            plt.close()

            # Evaluate
            p_real, p_fake = self.evaluate()
            test_p_real.append(p_real)
            test_p_fake.append(p_fake)
            
            # result of fixed noise input
            generate_images(f'Epoch {epoch+1}', 'images/', self.fixed_noise, 64, self.G, True)

            # print epoch result
            print(f'Epoch {epoch+1} -  train loss G: {train_loss_g[-1]:.4f} / train loss D: {train_loss_d[-1]:.4f}')
            print(f'           p_real: {p_real:.3f} / p_fake: {p_fake:.3f} ')
        
        # save model
        torch.save(self.G.state_dict(), f'models/generator epoch {epoch+1}.pth')
        torch.save(self.D.state_dict(), f'models/discriminator epoch {epoch+1}.pth')
        save_graph(train_loss_d, train_loss_g, test_p_real, test_p_fake, "graph.png")
        with open(f"history/train_history {epoch+1}.pkl", "wb") as f:
            pickle.dump([train_loss_d, train_loss_g, test_p_real, test_p_fake], f)
  
if __name__=='__main__':
    D_NOISE = 100
    NUM_G_CHANNEL = 256
    NUM_D_CHANNEL = 256
    NUM_IMG_CHANNEL = 1
    BATCH_SIZE = 128

    G = Generator(NUM_IMG_CHANNEL, D_NOISE, NUM_G_CHANNEL)
    D = Discriminator(NUM_IMG_CHANNEL, NUM_D_CHANNEL)
    
    train = Train(G, D, D_NOISE, BATCH_SIZE)
    train.train(num_epoch=20, k=1)