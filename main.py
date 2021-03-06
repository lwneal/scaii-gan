import argparse
import numpy as np
import os
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

import model
from logutil import TimeSeries
import imutil

device = torch.device("cuda")

print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--save_to_dir', type=str, default='checkpoints')
parser.add_argument('--load_from_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--latent_size', type=int, default=16)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--lambda_gan', type=float, default=0.1)
parser.add_argument('--dataset', type=str, required=True)

args = parser.parse_args()


from dataloader import CustomDataloader
from converter import SkyRTSConverter
loader = CustomDataloader(args.dataset, batch_size=args.batch_size, img_format=SkyRTSConverter)
test_loader = CustomDataloader(args.dataset, batch_size=args.batch_size, img_format=SkyRTSConverter, fold='test')


print('Building model...')

discriminator = model.Discriminator().to(device)
generator = model.Generator(args.latent_size).to(device)
encoder = model.Encoder(args.latent_size).to(device)
value_estimator = model.ValueEstimator(args.latent_size).to(device)
predictor = model.Predictor(args.latent_size).to(device)

if args.start_epoch:
    generator.load_state_dict(torch.load('{}/gen_{}'.format(args.load_from_dir, args.start_epoch)))
    encoder.load_state_dict(torch.load('{}/enc_{}'.format(args.load_from_dir, args.start_epoch)))
    discriminator.load_state_dict(torch.load('{}/disc_{}'.format(args.load_from_dir, args.start_epoch)))
    value_estimator.load_state_dict(torch.load('{}/value_{}'.format(args.load_from_dir, args.start_epoch)))
    value_estimator.load_state_dict(torch.load('{}/predictor_{}'.format(args.load_from_dir, args.start_epoch)))

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
print('Building optimizers')
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_gen_gan = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))
optim_class = optim.Adam(value_estimator.parameters(), lr=args.lr)
optim_predictor = optim.Adam(predictor.parameters(), lr=args.lr)

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_g_gan = optim.lr_scheduler.ExponentialLR(optim_gen_gan, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
scheduler_c = optim.lr_scheduler.ExponentialLR(optim_class, gamma=0.99)
scheduler_p = optim.lr_scheduler.ExponentialLR(optim_predictor, gamma=0.99)
print('Finished building model')


def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    z = normalize_vector(z)
    return z.to(device)


def normalize_vector(x, eps=.0001):
    norm = torch.norm(x, p=2, dim=1) + eps
    return x / norm.expand(1, -1).t()


def format_demo_img(state, qvals=None, caption_text="Title"):
    # Fill a white background
    canvas = np.ones((280, 140)) * 255
    canvas[32:72,  20:60] = state[0] * 255.
    canvas[92:132,  20:60] = state[1] * 255.
    canvas[152:192, 20:60] = state[2] * 255.

    canvas[32:72,  80:120] = state[3] * 255.
    canvas[92:132,  80:120] = state[4] * 255.
    canvas[152:192, 80:120] = state[5] * 255.

    # Now draw all the text captions
    from PIL import Image, ImageFont, ImageDraw
    img = Image.fromarray(canvas.astype('uint8'))
    # Should be available on Ubuntu 14.04+
    FONT_FILE = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font = ImageFont.truetype(FONT_FILE, 10)
    draw = ImageDraw.Draw(img)

    def draw_text(x, y, caption):
        textsize = draw.textsize(caption, font=font)
        #draw.rectangle([(x, y), textsize], fill=(0,))
        draw.multiline_text((x,y), caption, font=font)

    draw_text(0,0, caption_text)

    draw_text(20, 20, "Health")
    draw_text(20, 80, "Agent")
    draw_text(20, 140, "Small")

    draw_text(80, 20, "Large")
    draw_text(80, 80, "Friendly")
    draw_text(80, 140, "Enemy")

    if qvals is not None:
        draw_text(25, 207, "Reward Estimates")
        draw_text(10, 222, "Atk Top Left:  {:.2f}".format(qvals[3]))
        draw_text(10, 232, "Atk Top Right: {:.2f}".format(qvals[2]))
        draw_text(10, 242, "Atk Bot Left:  {:.2f}".format(qvals[1]))
        draw_text(10, 252, "Atk Bot Right: {:.2f}".format(qvals[0]))
    canvas = np.array(img)
    return canvas


def train(epoch, ts, max_batches=100, disc_iters=5):
    for i, (data, labels) in enumerate(islice(loader, max_batches)):
        current_frame = data[:, 0]
        next_frame = data[:, 1]
        qvals, mask = labels

        discriminator.train()
        encoder.eval()
        generator.eval()

        optim_disc.zero_grad()
        optim_gen.zero_grad()
        optim_enc.zero_grad()

        # Update discriminator
        z = sample_z(args.batch_size, args.latent_size)
        d_real = 1.0 - discriminator(current_frame)
        d_fake = 1.0 + discriminator(generator(z))
        disc_loss = nn.ReLU()(d_real).mean() + nn.ReLU()(d_fake).mean()
        ts.collect('Disc Loss', disc_loss)
        ts.collect('Disc (Real)', d_real.mean())
        ts.collect('Disc (Fake)', d_fake.mean())
        disc_loss.backward()
        optim_disc.step()

        encoder.train()
        generator.train()
        value_estimator.train()

        # Update generator (based on output of discriminator)
        optim_gen.zero_grad()
        z = sample_z(args.batch_size, args.latent_size)
        d_gen = 1.0 - discriminator(generator(z))
        gen_loss = nn.ReLU()(d_gen).mean()
        # Alternative: If you want to only make reconstructions realistic
        #d_gen = 1.0 - discriminator(generator(encoder(current_frame)))
        gen_loss.backward()
        optim_gen.step()

        # For Improved Wasserstein GAN:
        # gp_loss = calc_gradient_penalty(discriminator, ...)
        # gp_loss.backward()

        # Reconstruct pixels
        optim_enc.zero_grad()
        optim_gen.zero_grad()
        optim_class.zero_grad()
        optim_predictor.zero_grad()

        encoded = encoder(current_frame)
        reconstructed = generator(encoded)
        # Huber loss
        reconstruction_loss = F.smooth_l1_loss(reconstructed, current_frame)
        # Alternative: Good old-fashioned MSE loss
        #reconstruction_loss = torch.sum((reconstructed - current_frame)**2)
        ts.collect('Reconst Loss', reconstruction_loss)
        ts.collect('Z variance', encoded.var(0).mean())
        ts.collect('Reconst Pixel variance', reconstructed.var(0).mean())
        ts.collect('Z[0] mean', encoded[:,0].mean().item())

        # ValueEstimator outputs linear scores (logits)
        predictions = value_estimator(encoder(current_frame))

        # MSE loss, but only for the available data
        qloss = torch.mean(mask * ((qvals - predictions) **2))
        ts.collect('Q Value Regression Loss', qloss)

        # Reconstruction loss for the simulated next frame
        # But again, ~only~ for the frames we know the future state of
        predicted_successors = predictor(encoded)
        known_idx = mask.max(dim=1)[1]
        indices = known_idx.view(-1,1,1).expand(args.batch_size,1,args.latent_size)
        predicted_latent_points = predicted_successors.gather(1, indices)
        predicted_next_frame = generator(predicted_latent_points)

        pred_rec_loss = F.smooth_l1_loss(predicted_next_frame, next_frame)
        ts.collect('Pred Recon Loss', pred_rec_loss)

        loss = reconstruction_loss + qloss + pred_rec_loss
        loss.backward()

        optim_class.step()
        optim_enc.step()
        optim_gen.step()
        optim_predictor.step()

        if i % 5 == 0:
            # GAN update for realism
            optim_gen_gan.zero_grad()
            z = sample_z(args.batch_size, args.latent_size)
            generated = generator(z)
            gen_loss = -discriminator(generated).mean() * args.lambda_gan
            ts.collect('Generated pixel variance', generated.var(0).mean())
            ts.collect('Gen Loss', gen_loss)
            gen_loss.backward()
            optim_gen_gan.step()

        ts.print_every(n_sec=4)

    scheduler_e.step()
    scheduler_d.step()
    scheduler_g.step()
    scheduler_g_gan.step()
    print(ts)


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def make_counterfactual_trajectory(x, target_action, iters=300, initial_speed=0.1,
                                   speed_decay=0.99, mu=0.9, stability_coefficient=1.0):
    trajectory = []

    z0 = encoder(x)[0]
    z = z0.clone()
    original_qvals = value_estimator(z0)
    losses = []

    speed = initial_speed
    velocity = torch.zeros(z.size()).to(device)

    for i in range(iters):
        cf_loss = 0
        qvals = value_estimator(z)
        for class_idx in range(len(qvals)):
            if class_idx == target_action:
                cf_loss += (1 - qvals[class_idx]) ** 2
            else:
                cf_loss += stability_coefficient * (qvals[class_idx] - original_qvals[class_idx])**2

        dc_dz = autograd.grad(cf_loss, z, cf_loss)[0]
        losses.append(float(cf_loss))

        v_prev = velocity
        velocity = mu * velocity - speed * dc_dz
        z += -mu * v_prev + (1 + mu) * velocity
        z /= torch.norm(z)
        speed *= speed_decay
        trajectory.append([to_np(z)])

    distance = float(torch.norm(z - z0, p=2))
    print('Counterfactual distance {:.3f} initial loss {:.3f} final loss {:.3f}'.format(
        distance, losses[0], losses[-1]))
    return np.array(trajectory)


def make_video(output_video_name, trajectory, whatif=""):
    print('Generating video from trajectory shape {}'.format(trajectory.shape))
    generator.eval()
    v = imutil.VideoMaker(output_video_name)

    z_0 = torch.Tensor(trajectory[0]).to(device)
    original_samples = generator(z_0)[0]
    original_qvals = value_estimator(z_0)[0]
    left_pixels = format_demo_img(to_np(original_samples), to_np(original_qvals),
                                  'Reality')
    for z in torch.Tensor(trajectory):
        z = z.to(device)
        samples = generator(z)[0]
        qvals = value_estimator(z)[0]
        right_pixels = format_demo_img(to_np(samples), to_np(qvals),
                                       'What If: {}'.format(whatif))
        pixels = np.concatenate([left_pixels, right_pixels], axis=1)
        v.write_frame(pixels)
    v.finish()


def main():
    os.makedirs(args.save_to_dir, exist_ok=True)
    batches_per_epoch = 200
    ts_train = TimeSeries('Training', batches_per_epoch * args.epochs)
    for epoch in range(args.epochs):
        print('starting epoch {}'.format(epoch))
        train(epoch, ts_train, batches_per_epoch)
        print(ts_train)

        data, _ = next(i for i in test_loader)
        for target_action in range(4):
            curr_frame = data[:,0]
            cf_trajectory = make_counterfactual_trajectory(curr_frame, target_action)
            filename = 'cf_epoch_{:03d}_{}'.format(epoch, target_action)
            make_video(filename, cf_trajectory, whatif=' action={}'.format(target_action))

    torch.save(discriminator.state_dict(), os.path.join(args.save_to_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.save_to_dir, 'gen_{}'.format(epoch)))
    torch.save(encoder.state_dict(), os.path.join(args.save_to_dir, 'enc_{}'.format(epoch)))
    torch.save(value_estimator.state_dict(), os.path.join(args.save_to_dir, 'value_{}'.format(epoch)))
    torch.save(predictor.state_dict(), os.path.join(args.save_to_dir, 'predictor_{}'.format(epoch)))


if __name__ == '__main__':
    main()
