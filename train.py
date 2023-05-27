import torch
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Generator, Discriminator
from utils import *


class Train():
    def __init__(self, generator, discriminator, d_noise):
        self.G = generator
        self.D = discriminator
        he_initialization(self.G.parameters(), 'relu')
        he_initialization(self.D.parameters(), 'leaky_relu', 0)

        self.optim_g = optim.Adam(self.G.parameters(), lr = 0.0002)
        self.optim_d = optim.Adam(self.D.parameters(), lr = 0.0002)
        self.d_noise = d_noise
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G.to(self.device)
        self.D.to(self.device)

        self.dataloader()

        self.fixed_noise = torch.randn(64, 100, 1, 1, device=self.device)

    def sample_z(self, batch_size):
        return torch.randn(batch_size, self.d_noise, device=self.device).unsqueeze(-1).unsqueeze(-1)

    def dataloader(self):
        standardizator = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5),  
                                                        std=(0.5))])

        train_data = dsets.MNIST(root='data/', train=True, transform=standardizator, download=True)
        test_data = dsets.MNIST(root='data/', train=False, transform=standardizator, download=True)

        self.batch_size = 256
        self.train_data_loader = DataLoader(train_data, self.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(test_data, self.batch_size, shuffle=True)
        print("finished getting data")

    def run_epoch_G(self):
        train_loss_g = 0
            
        for train_img, train_label in tqdm(self.train_data_loader):
            self.optim_g.zero_grad()

            noise = self.sample_z(self.batch_size if len(train_img)==self.batch_size else len(train_img))
            p_fake = self.D(self.G(noise))

            loss_g = -1 * torch.log(p_fake).mean()
            loss_g.backward()
            self.optim_g.step()

            train_loss_g += loss_g
            break

        return train_loss_g / len(self.train_data_loader)
    
    def run_epoch_D(self):
        train_loss_d = 0

        for train_img, train_label in tqdm(self.train_data_loader):
            train_img, train_label = train_img.to(self.device), train_label.to(self.device)
            
            self.optim_d.zero_grad()
            self.optim_g.zero_grad()

            p_real = self.D(train_img)
            noise = self.sample_z(self.batch_size if len(train_img)==self.batch_size else len(train_img))
            p_fake = self.D(self.G(noise))

            loss_real = -1 * torch.log(p_real)
            loss_fake = -1 * torch.log(1 - p_fake)
            loss_d = (loss_real + loss_fake).mean()
            loss_d.backward()
            self.optim_d.step()

            train_loss_d += loss_d
            break

        return train_loss_d / len(self.train_data_loader)

    def evaluate(self):
        p_real, p_fake = 0., 0.
        for test_img, test_label in self.test_data_loader:
            test_img, test_label = test_img.to(self.device), test_label.to(self.device)

            with torch.autograd.no_grad():
                p_real += (torch.sum(self.D(test_img)).item())
                noise = self.sample_z(self.batch_size if len(test_img)==self.batch_size else len(test_img))
                p_fake += (torch.sum(self.D(self.G(noise))).item())
            break
            
        return p_real / len(self.test_data_loader.dataset), p_fake / len(self.test_data_loader.dataset)

    def train(self, num_epoch, k):
        train_loss_d_list = []
        train_loss_g_list = []
        p_real_list = []
        p_fake_list = []
        
        for epoch in range(1, num_epoch+1):
            print(f"Epoch {epoch}")
            for _ in range(k):
                train_loss_d = self.run_epoch_D()
            train_loss_g = self.run_epoch_G()

            p_real, p_fake = self.evaluate()

            train_loss_d_list.append(train_loss_d)
            train_loss_g_list.append(train_loss_g)
            p_real_list.append(p_real)
            p_fake_list.append(p_fake)
            
            generate_images(epoch, 'images/', self.fixed_noise, 64, self.G)
        
        # save
        torch.save(self.G.state_dict(), f'generator epoch {epoch}.pth')
        torch.save(self.D.state_dict(), f'discriminator epoch {epoch}.pth')
        save_graph(train_loss_d_list, train_loss_g_list, p_real_list, p_fake_list, "graph.png")
        
        return train_loss_d_list, train_loss_g_list, p_real_list, p_fake_list


if __name__=='__main__':
    nz = 100
    ngf = 256
    ndf = 256
    nc = 1
    G = Generator(nc, nz, ngf)
    D = Discriminator(nc, ndf)
    train = Train(G, D, nz)
    train.train(num_epoch=100, k=2)