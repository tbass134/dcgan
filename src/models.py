import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision


# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()

        self.device = device
        netG = Generator(ngpu).to(device)
        self.netG = netG

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netG.apply(self.weights_init)

        # Print the model
        print(self.netG)

        # Create the Discriminator
        netD = Discriminator(ngpu).to(device)
        self.netD = netD

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        self.netD.apply(self.weights_init)

        # Print the model
        print(self.netD)

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_on_dataset(self,dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        lossD = []
        lossG = []
        for i, data in enumerate(dataloader, 0):
            
            self.netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
            # Forward pass real batch through D
            output = self.netD(real_cpu).view(-1)
            # Calculate      loss on all-real batch
            errD_real = self.criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=self.device)
            # Generate fake image batch with G
            fake = self.netG(noise)
            label.fill_(self.fake_label)
            # Classify all fake batch with D
            output = self.netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(self.real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()
            lossD.append(errD.item())
            lossG.append(errG.item())


            if i % 50 == 0:
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        return lossD, lossG

    @torch.no_grad()
    def vis_on_dataset(self,fname):
        fake = self.netG(self.fixed_noise).detach().cpu()
        # grid = vutils.make_grid(fake, padding=2, normalize=True)
        new_img =  torchvision.utils.make_grid(fake).numpy().transpose(1,2,0)
        plt.axis('off')
        plt.imshow(new_img)
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



