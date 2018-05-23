import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import model

import numpy as np
import os
from gnomehat import imutil


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--latent_size', type=int, default=10)
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--env_name', type=str, default='Pong-v0')

args = parser.parse_args()


print('Initializing dataloader')
from dataloader import CustomDataloader
from converter import SkyRTSConverter
loader = CustomDataloader('/mnt/nfs/data/towers.dataset', img_format=SkyRTSConverter)


print('Building model...')
Z_dim = args.latent_size
#number of updates to discriminator for every update to generator
disc_iters = 5

discriminator = model.Discriminator().cuda()
generator = model.Generator(Z_dim).cuda()
encoder = model.Encoder(Z_dim).cuda()

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
print('finished building model')

def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    return z.cuda()

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        print('training batch {}'.format(batch_idx))
        if data.size()[0] != args.batch_size:
            continue
        data = data.cuda()

        # reconstruct images
        optim_enc.zero_grad()
        optim_gen.zero_grad()

        reconstructed = generator(encoder(data))

        aac_loss = torch.sum((reconstructed - data)**2)
        aac_loss.backward()

        optim_enc.step()
        optim_gen.step()

        # update discriminator
        for _ in range(disc_iters):
            z = sample_z(args.batch_size, Z_dim)
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            disc_loss.backward()
            optim_disc.step()

        z = sample_z(args.batch_size, Z_dim)

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        gen_loss = -discriminator(generator(z)).mean()
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 10 == 0:
            print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
            print("Losses:  AAC: {:.3f}  D {:.3f}  G {:.3f}".format(
                aac_loss.data[0], disc_loss.data[0], gen_loss.data[0]))
    scheduler_e.step()
    scheduler_d.step()
    scheduler_g.step()


fixed_z = sample_z(args.batch_size, Z_dim)
def evaluate(epoch):
    samples = generator(fixed_z).cpu().data.numpy()[:64]
    scores = []
    for i in range(10):
        scores.append(evaluate_fit(epoch, i))
    avg_mse = np.array(scores).mean()
    print("Epoch {} Avg Encoding MSE:{:.4f}".format(epoch, avg_mse))


def evaluate_fit(epoch, idx=0):
    # Get a random Atari frame, try to fit it by gradient descent
    frames, _ = next(loader.__iter__())
    frame = frames[idx]
    frame = Variable(frame.cuda())

    # TODO: Instead of standard gradient descent, this should be
    #  projected gradient descent on eg. the unit sphere if the
    #  behavior of sample_z is changed
    z = Variable(torch.randn(1, Z_dim).cuda(), requires_grad=True)

    speed = .01
    for _ in range(100):
        encoded = generator(z)[0]
        mse = (frame - encoded) ** 2
        loss = mse.sum()
        df_dz = autograd.grad(loss, z, loss)[0]
        z = z - speed * df_dz
        speed *= .99  # annealing schedule

    if idx == 0:
        filename = 'fit_{:03d}_{:04d}.png'.format(epoch, idx)
        comparison = torch.cat((frame.expand(1,-1,-1,-1), encoded.expand((1, -1, -1, -1))))
        imutil.show(comparison, filename=filename)
    return loss.data[0]


def encode(frame):
    speed = .01
    z = Variable(torch.randn(1, Z_dim).cuda(), requires_grad=True)
    for _ in range(100):
        encoded = generator(z)[0]
        mse = (frame - encoded) ** 2
        loss = mse.sum()
        df_dz = autograd.grad(loss, z, loss)[0]
        z = z - speed * df_dz
        speed *= .99  # annealing schedule
    return encoded



fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
fixed_zprime = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def make_video(output_video_name):
    v = imutil.VideoMaker(output_video_name)
    for i in range(400):
        theta = abs(i - 200) / 200.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        #z = z[:args.batch_size]
        samples = generator(z).cpu().data.numpy()
        pixels = samples.transpose((0,2,3,1)) * 0.5 + 0.5
        v.write_frame(pixels)
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    for epoch in range(args.epochs):
        print('starting epoch {}'.format(epoch))
        train(epoch)
        #make_video('epoch_{:03d}'.format(epoch))
        evaluate(epoch)
        torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))


if __name__ == '__main__':
    main()
