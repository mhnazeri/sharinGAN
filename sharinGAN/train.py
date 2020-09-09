import numpy as np
import torch
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import get_device, plot_images, weights_init, get_conf
from model.data_loader import SharinganDataset
from model.net import Discriminator, Generator


def main():
    cfg = get_conf("conf/train")
    device = get_device()
    # Create the generator
    netG = Generator().to(device)

    # Handle multi-gpu
    if (device.type == 'cuda') and (cfg.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(cfg.ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(init_weights)
    # Create the Discriminator
    netD = Discriminator().to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (cfg.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(cfg.ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(init_weights)
    # transform images
    transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = SharinganDataset(transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    batch = next(iter(dataloader))
    plot_images(batch)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    fixed_noise = torch.randn(cfg.batch_size, cfg.nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    # load adam optimizer's config
    cfg_adam = get_conf("conf/optimizer/adam")
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=cfg_adam.lr, betas=(cfg_adam.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg_adam.lr, betas=(cfg_adam.beta1, 0.999))
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(cfg.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, cfg.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # alternative loss with bigger gradients
            # errD_fake = criterion(output, label) + (cfg.beta * (fake.detach() - real_cpu).pow(2).mean(0).sum())
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # grad clip
            if cfg.clipping_threshold_d > 0:
                nn.utils.clip_grad_norm_(netD.parameters(), cfg.clipping_threshold_d)
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 25 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, cfg.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == cfg.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # plot discriminator and generator losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    # plot a real images alongside generated ones
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:25], padding=5, normalize=True).cpu(), (1, 2, 0)))
    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()



if __name__ == "__main__":
    main()
