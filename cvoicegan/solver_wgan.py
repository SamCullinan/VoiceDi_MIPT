from model_wgan import Generator
from model_wgan import Discriminator
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, config):
        """Initialize configurations."""

        # Data loader.
        self.loader = loader

        # Model configurations.
        self.lambda_gp = config.lambda_gp
        self.lambda_cls = config.lambda_cls
        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        self.dim_z = config.dim_z
        self.dim_h = config.dim_h
        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(dim_z=self.dim_z, dim_h=self.dim_h)
        self.D = Discriminator(dim_h=self.dim_h)

        self.g_optimizer = torch.optim.Adam(
            self.G.parameters(), self.g_lr, [self.beta1, self.beta2]
        )
        self.d_optimizer = torch.optim.Adam(
            self.D.parameters(), self.d_lr, [self.beta1, self.beta2]
        )
        self.print_network(self.G, "G")
        self.print_network(self.D, "D")

        self.G.to(self.device).train()
        self.D.to(self.device).train()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print("Loading the trained models from step {}...".format(resume_iters))
        G_path = os.path.join(self.model_save_dir, "{}-G.ckpt".format(resume_iters))
        D_path = os.path.join(self.model_save_dir, "{}-D.ckpt".format(resume_iters))
        self.G.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage)
        )
        self.D.load_state_dict(
            torch.load(D_path, map_location=lambda storage, loc: storage)
        )

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter

        self.logger = SummaryWriter(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print("Start training...")
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            try:
                x_real, y_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, y_org = next(data_iter)

            rand_idx = torch.randperm(y_org.size(0))
            y_trg = y_org[rand_idx]

            x_real = x_real.to(self.device)
            y_org = y_org.to(self.device)
            y_trg = y_trg.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss for real samples
            out_src, out_cls = self.D(x_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = F.cross_entropy(out_cls, y_org)

            # Compute loss for fake samples
            z = torch.randn(x_real.size(0), self.dim_z, device=self.device)
            x_fake = self.G(z, y_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(
                True
            )
            out_src, out_cls = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = (
                d_loss_real
                + d_loss_fake
                + self.lambda_gp * d_loss_gp
                + self.lambda_cls * d_loss_cls
            )
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss["D/loss_real"] = d_loss_real.item()
            loss["D/loss_fake"] = d_loss_fake.item()
            loss["D/loss_cls"] = d_loss_cls.item()
            loss["D/loss_gp"] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i + 1) % self.n_critic == 0:

                z = torch.randn(x_real.size(0), self.dim_z, device=self.device)
                x_fake = self.G(z, y_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = F.cross_entropy(out_cls, y_trg)

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss["G/loss_fake"] = g_loss_fake.item()
                loss["G/loss_cls"] = g_loss_cls.item()
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i + 1, self.num_iters
                )
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                if (i + 1) % 100 == 0:
                    print("tensorboard --logdir {}".format(self.log_dir))
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.add_scalar(tag, value, i + 1)

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    z = torch.randn(16, self.dim_z, device=self.device)

                    x_sample = self.G(z, torch.zeros(16)).detach().cpu().numpy()
                    sample_path = os.path.join(
                        self.sample_dir, "{}-M-samples".format(i + 1)
                    )
                    np.save(
                        sample_path, x_sample.astype(np.float32), allow_pickle=False
                    )

                    x_sample = self.G(z, torch.ones(16)).detach().cpu().numpy()
                    sample_path = os.path.join(
                        self.sample_dir, "{}-F-samples".format(i + 1)
                    )
                    np.save(
                        sample_path, x_sample.astype(np.float32), allow_pickle=False
                    )

            # Save model checkpoints.
            if (i + 1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, "{}-G.ckpt".format(i + 1))
                D_path = os.path.join(self.model_save_dir, "{}-D.ckpt".format(i + 1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print("Saved model checkpoints into {}...".format(self.model_save_dir))

            # Decay learning rates.
            if (i + 1) % self.lr_update_step == 0 and (i + 1) > (
                self.num_iters - self.num_iters_decay
            ):
                g_lr -= self.g_lr / float(self.num_iters_decay)
                d_lr -= self.d_lr / float(self.num_iters_decay)
                self.update_lr(g_lr, d_lr)
                print("Decayed learning rates, g_lr: {}, d_lr: {}.".format(g_lr, d_lr))
