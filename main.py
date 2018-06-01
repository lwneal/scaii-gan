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
parser.add_argument('--save_to_dir', type=str, default='checkpoints')
parser.add_argument('--load_from_dir', type=str, default='checkpoints')
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
classifier = model.Classifier(Z_dim).to(device)

if args.start_epoch:
    generator.load_state_dict(torch.load('{}/gen_{}'.format(args.load_from_dir, args.start_epoch)))
    encoder.load_state_dict(torch.load('{}/enc_{}'.format(args.load_from_dir, args.start_epoch)))
    discriminator.load_state_dict(torch.load('{}/disc_{}'.format(args.load_from_dir, args.start_epoch)))
    classifier.load_state_dict(torch.load('{}/class_{}'.format(args.load_from_dir, args.start_epoch)))

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
print('Building optimizers')
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_gen_gan = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_class = optim.Adam(classifier.parameters(), lr=args.lr)

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_g_gan = optim.lr_scheduler.ExponentialLR(optim_gen_gan, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
scheduler_c = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
print('Finished building model')


def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    z = normalize_vector(z)
    return z.to(device)


def normalize_vector(x, eps=.0001):
    norm = torch.norm(x, p=2, dim=1) + eps
    return x / norm.expand(1, -1).t()


def format_demo_img(state):
    # Fill a white background
    canvas = np.ones((190, 140)) * 255
    canvas[20:60,  20:60] = state[0] * 255.
    canvas[80:120,  20:60] = state[1] * 255.
    canvas[140:180, 20:60] = state[2] * 255.

    canvas[20:60,  80:120] = state[3] * 255.
    canvas[80:120,  80:120] = state[4] * 255.
    canvas[140:180, 80:120] = state[5] * 255.

    # Now draw all the text captions
    from PIL import Image, ImageFont, ImageDraw
    img = Image.fromarray(canvas.astype('uint8'))
    FONT_FILE = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font = ImageFont.truetype(FONT_FILE, 10)
    draw = ImageDraw.Draw(img)

    def draw_text(x, y, caption):
        textsize = draw.textsize(caption, font=font)
        #draw.rectangle([(x, y), textsize], fill=(0,0,0,128))
        draw.multiline_text((x,y), caption, font=font)

    draw_text(20, 8, "Health")
    draw_text(20, 68, "Agent")
    draw_text(20, 128, "Small")

    draw_text(80, 8, "Large")
    draw_text(80, 68, "Friendly")
    draw_text(80, 128, "Enemy")
    canvas = np.array(img)
    return canvas


def train(epoch, ts, max_batches=100, disc_iters=5):

    for i, (data, labels) in enumerate(loader):
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
        classifier.train()
        optim_enc.zero_grad()
        optim_gen.zero_grad()
        optim_class.zero_grad()

        # reconstruct images
        encoded = encoder(data)
        reconstructed = generator(encoded)
        #reconstruction_loss = torch.mean((reconstructed - data)**2)
        reconstruction_loss = F.smooth_l1_loss(reconstructed, data)
        reconstruction_loss.backward()
        ts.collect('Reconst. Loss', reconstruction_loss)
        ts.collect('Z variance', encoded.var(0).mean())
        ts.collect('Reconst. Pixel variance', reconstructed.var(0).mean())
        ts.collect('Z[0] mean', encoded[:,0].mean().item())

        # Update classifier
        preds = classifier(encoded)
        nll_loss = F.nll_loss(preds, labels)
        ts.collect('Classifier NLL', nll_loss)

        optim_enc.step()
        optim_gen.step()
        optim_class.step()

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

        ts.print_every(n_sec=4)
        if i % 100 == 0:
            img_real = format_demo_img(to_np(data[0]))
            img_recon = format_demo_img(to_np(reconstructed[0]))
            demo_img = np.concatenate([img_real, img_recon], axis=1)
            filename = 'reconstruction_epoch{:03d}_{:04d}.png'.format(epoch, i)
            imutil.show(demo_img, filename=filename)
            generated = generator(sample_z(1, Z_dim))
            gen_img = format_demo_img(to_np(generated[0]))
            filename = 'generated_epoch{:03d}_{:04d}.png'.format(epoch, i)
            imutil.show(gen_img, filename=filename)

    scheduler_e.step()
    scheduler_d.step()
    scheduler_g.step()
    scheduler_g_gan.step()
    print(ts)


def to_np(tensor):
    return tensor.detach().cpu().numpy()


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
        reconstructions.append(to_np(real_images[i]).transpose(1,2,0))
        reconstructions.append(to_np(reconstructed[i]).transpose(1,2,0))
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
    for i in range(200):
        theta = abs(i - 100) / 100.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        z = normalize_vector(z)
        samples = generator(z)
        pixels = format_demo_img(to_np(samples[0]))
        v.write_frame(pixels, normalize_color=False, resize_to=(380,280))
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.save_to_dir, exist_ok=True)
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

        torch.save(discriminator.state_dict(), os.path.join(args.save_to_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.save_to_dir, 'gen_{}'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.save_to_dir, 'enc_{}'.format(epoch)))
        torch.save(classifier.state_dict(), os.path.join(args.save_to_dir, 'class_{}'.format(epoch)))


if __name__ == '__main__':
    main()
