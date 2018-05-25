import argparse
import numpy as np
import os
import random
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch import autograd

import model
from gnomehat.series import TimeSeries
from gnomehat import imutil

device = torch.device("cuda")

print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--latent_size', type=int, default=4)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lambda_gan', type=float, default=0.1)
parser.add_argument('--dataset', type=str, required=True)

args = parser.parse_args()
Z_dim = args.latent_size


from dataloader import CustomDataloader
from converter import SkyRTSConverter
loader = CustomDataloader(args.dataset, batch_size=args.batch_size, img_format=SkyRTSConverter)


print('Building model...')

discriminator = model.Discriminator().to(device)
generator = model.Generator(Z_dim).to(device)
encoder = model.Encoder(Z_dim).to(device)

if args.start_epoch:
    generator.load_state_dict(torch.load('checkpoints/gen_{}'.format(args.start_epoch)))
    encoder.load_state_dict(torch.load('checkpoints/enc_{}'.format(args.start_epoch)))
    discriminator.load_state_dict(torch.load('checkpoints/disc_{}'.format(args.start_epoch)))

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
print('Building optimizers')
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_gen_gan = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_g_gan = optim.lr_scheduler.ExponentialLR(optim_gen_gan, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
print('Finished building model')


def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    return z.to(device)


def train(epoch, ts, max_batches=100, disc_iters=5):

    for data, labels in loader:
        # update discriminator
        discriminator.train()
        encoder.eval()
        generator.eval()

        z = sample_z(args.batch_size, Z_dim)
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        optim_enc.zero_grad()
        d_real = 1.0 - discriminator(data)
        d_fake = 1.0 + discriminator(generator(z))
        disc_loss = nn.ReLU()(d_real).mean() + nn.ReLU()(d_fake).mean()
        disc_loss.backward()
        optim_disc.step()
        ts.collect('Disc (Real)', d_real.mean())
        ts.collect('Disc (Fake)', d_fake.mean())

        encoder.train()
        generator.train()
        optim_enc.zero_grad()
        optim_gen.zero_grad()

        # reconstruct images
        encoded = encoder(data)
        reconstructed = generator(encoded)
        #reconstruction_loss = torch.mean((reconstructed - data)**2)
        reconstruction_loss = F.smooth_l1_loss(reconstructed, data)

        reconstruction_loss.backward()
        ts.collect('Z variance', encoded.var(0).mean())
        ts.collect('Reconst. Pixel variance', reconstructed.var(0).mean())
        ts.collect('Z[0] mean', encoded[:,0].mean().item())

        optim_enc.step()
        optim_gen.step()

        # GAN update for realism
        optim_gen_gan.zero_grad()
        z = sample_z(args.batch_size, Z_dim)
        generated = generator(z)
        gen_loss = -discriminator(generated).mean() * args.lambda_gan
        ts.collect('Generated pixel variance', generated.var(0).mean())
        gen_loss.backward()
        optim_gen_gan.step()

        ts.collect('Disc Loss', disc_loss)
        ts.collect('Gen Loss', gen_loss)
        ts.collect('Reconst. Loss', reconstruction_loss)

        ts.print_every(n_sec=4)

    # Reconstruct real frames
    reconstructed = generator(encoder(data))

    reconstructions = []
    for i in range(8):
        reconstructions.append(to_np(data[i]))
        reconstructions.append(to_np(reconstructed[i]))
    img_recon = np.array(reconstructions)
    # HACK: Show only the first 3 channels
    img_recon = img_recon[:,:,:,:3]
    imutil.show(img_recon, filename='training_reconstruction_epoch_{:05d}.png'.format(epoch))

    scheduler_e.step()
    scheduler_d.step()
    scheduler_g.step()
    scheduler_g_gan.step()
    print(ts)


def to_np(x):
    return x.detach().cpu().numpy().transpose(1,2,0)


fixed_z = sample_z(args.batch_size, Z_dim)
def evaluate(epoch, img_samples=8):
    encoder.eval()
    generator.eval()
    discriminator.eval()

    samples = generator(fixed_z).cpu().data.numpy()

    eval_loader = AtariDataloader(batch_size=args.batch_size)
    real_images, _ = next(eval_loader)
    real_images = real_images.to(device)

    # Reconstruct real frames
    reconstructed = generator(encoder(real_images))
    reconstruction_l2 = torch.mean((reconstructed - real_images) ** 2).item()

    reconstructions = []
    for i in range(img_samples):
        reconstructions.append(to_np(real_images[i]))
        reconstructions.append(to_np(reconstructed[i]))
    img_recon = np.array(reconstructions)
    imutil.show(img_recon, filename='reconstruction_epoch_{:05d}.png'.format(epoch))

    # Reconstruct generated frames
    reconstructed = generator(encoder(generator(fixed_z)))
    cycle_reconstruction_l2 = torch.mean((real_images - reconstructed) ** 2).item()

    # TODO: Measure "goodness" of generated images

    # TODO: Measure disentanglement of latent codes

    # Generate images
    img = samples[:img_samples].transpose((0,2,3,1))
    imutil.show(img, filename='gen_epoch_{:05d}.png'.format(epoch))

    return {
        'generated_pixel_mean': samples.mean(),
        'generated_pixel_variance': samples.var(0).mean(),
        'reconstruction_l2': reconstruction_l2,
        'cycle_reconstruction_l2': cycle_reconstruction_l2,
    }


def make_video(output_video_name):
    generator.eval()
    fixed_z = torch.randn(1, Z_dim).to(device)
    fixed_zprime = torch.randn(1, Z_dim).to(device)
    v = imutil.VideoMaker(output_video_name)
    for i in range(100):
        theta = abs(i - 50) / 50.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        samples = generator(z).cpu().data.numpy()
        pixels = samples.transpose((0,2,3,1)) * 255
        v.write_frame(pixels, normalize_color=False)
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    batches_per_epoch = 100
    ts_train = TimeSeries('Training', batches_per_epoch * args.epochs)
    ts_eval = TimeSeries('Evaluation', args.epochs)
    for epoch in range(args.epochs):
        print('starting epoch {}'.format(epoch))
        train(epoch, ts_train, batches_per_epoch)
        print(ts_train)

        """
        metrics = evaluate(epoch)
        for key, value in metrics.items():
            ts_eval.collect(key, value)
        print(ts_eval)
        """
        make_video('epoch_{:03d}'.format(epoch))

        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'enc_{}'.format(epoch)))


if __name__ == '__main__':
    main()
