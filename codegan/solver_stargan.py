from model_stargan import Generator
from model_stargan import Discriminator
from librosa.output import write_wav
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
import datetime
from torch.utils import data
import os
import sys

sys.path.insert(1, "/vol/bitbucket/apg416/project/utils")
from decomposition import Decomposition

sys.path.insert(1, "/vol/bitbucket/apg416/project/waveglow")


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, loader, config):
        """Initialize configurations."""

        # Data loader.
        self.loader = loader

        self.class_weight = loader.dataset.class_weight

        # Model configurations.
        self.c_dim = config.c_dim
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_id = config.lambda_id

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

        # Test configurations.
        self.test_iters = config.test_iters

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

        self.code = config.code
        self.code_dim = {"pitch": 64, "rhythm": 2, "content": 16, "pitch_rhythm": 66}[
            config.code
        ]
        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""

        self.G = Generator(self.c_dim, self.code_dim)
        self.D = Discriminator(self.c_dim, self.code_dim)
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

        self.SS = Decomposition(
            "/vol/bitbucket/apg416/project/decomposition/run_full/models/2800000-G.ckpt",
            freeze=True,
        )
        self.SS.to(self.device)

        waveglow = torch.load(
            "/vol/bitbucket/apg416/project/waveglow/checkpoints/waveglow_128000"
        )["model"]
        self.WG = waveglow.remove_weightnorm(waveglow)
        self.WG.to(self.device).eval()

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
        dydx = dydx.reshape(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim):
        """Generate target domain labels for debugging and testing."""
        c_trg_list = []
        for i in range(c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(
            logit, target, weight=self.class_weight.float().to(self.device)
        )

    def get_input(self, codes_x, codes_f0, codes_2):
        if self.code == "content":
            return codes_x
        elif self.code == "pitch":
            return codes_f0
        elif self.code == "rhythm":
            return codes_2
        elif self.code == "pitch_rhythm":
            return torch.cat([codes_f0, codes_2], dim=2)

    def get_output(self, output, codes_x, codes_f0, codes_2):
        if self.code == "content":
            return output, codes_f0, codes_2
        elif self.code == "pitch":
            return codes_x, output, codes_2
        elif self.code == "rhythm":
            return codes_x, codes_f0, output
        elif self.code == "pitch_rhythm":
            return codes_x, output[:, :, :64], output[:, :, 64:]

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, emb_fixed, f0_fixed, _, c_org = next(iter(self.loader))

        x_fixed = x_fixed.to(self.device)
        emb_fixed = emb_fixed.to(self.device)
        f0_fixed = f0_fixed.to(self.device)

        C_fixed, emb_fixed, codes_x_fixed, codes_f0_fixed, codes_2_fixed = self.SS(
            x_fixed, emb_fixed, f0_fixed
        )

        c_fixed_list = self.create_labels(c_org, self.c_dim)

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
                x_org, emb_org, f0_org, _, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_org, emb_org, f0_org, _, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim)
            c_trg = self.label2onehot(label_trg, self.c_dim)

            c_org = c_org.to(self.device)  # Original domain labels.
            c_trg = c_trg.to(self.device)  # Target domain labels.
            label_org = label_org.to(
                self.device
            )  # Labels for computing classification loss.
            label_trg = label_trg.to(
                self.device
            )  # Labels for computing classification loss.

            x_org = x_org.to(self.device)
            emb_org = emb_org.to(self.device)
            f0_org = f0_org.to(self.device)

            C_org, emb_org, codes_x_org, codes_f0_org, codes_2_org = self.SS(
                x_org, emb_org, f0_org
            )

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            codes_real = self.get_input(codes_x_org, codes_f0_org, codes_2_org)

            # Compute loss with real images.
            out_src, out_cls = self.D(codes_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            # Compute loss with fake images.
            codes_fake = self.G(codes_real, c_trg)
            out_src, out_cls = self.D(codes_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(codes_real.size(0), 1, 1).to(self.device)
            codes_hat = (
                alpha * codes_real.data + (1 - alpha) * codes_fake.data
            ).requires_grad_(True)

            out_src, _ = self.D(codes_hat)

            d_loss_gp = self.gradient_penalty(out_src, codes_hat)
            # Backward and optimize.
            d_loss = (
                d_loss_real
                + d_loss_fake
                + self.lambda_cls * d_loss_cls
                + self.lambda_gp * d_loss_gp
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
                # Original-to-target domain.
                codes_fake = self.G(codes_real, c_trg)

                out_src, out_cls = self.D(codes_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                # Target-to-original domain.
                codes_reconst = self.G(codes_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(codes_real - codes_reconst))

                # Backward and optimize.
                g_loss = (
                    g_loss_fake
                    + self.lambda_cls * g_loss_cls
                    + self.lambda_rec * g_loss_rec
                )
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss["G/loss_fake"] = g_loss_fake.item()
                loss["G/loss_rec"] = g_loss_rec.item()
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
                    x_fake_list = [x_fixed]

                    for c_fixed in c_fixed_list:
                        output_fixed = self.G(
                            self.get_input(
                                codes_x_fixed, codes_f0_fixed, codes_2_fixed
                            ),
                            c_fixed,
                        )
                        (
                            codes_sample_x,
                            codes_sample_f0,
                            codes_sample_2,
                        ) = self.get_output(
                            output_fixed, codes_x_fixed, codes_f0_fixed, codes_2_fixed
                        )

                        x_fake_list.append(
                            self.SS.decode(
                                C_fixed,
                                emb_fixed,
                                codes_sample_x,
                                codes_sample_f0,
                                codes_sample_2,
                            )
                        )

                    x_concat = torch.cat(x_fake_list, dim=1)

                    for j, mel in enumerate(x_concat.unbind()):
                        audio = (
                            self.WG.infer(mel.unsqueeze(0).transpose(2, 1), sigma=1)
                            .detach()
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )
                        sample_path = os.path.join(
                            self.sample_dir, "I{}E{}-samples.wav".format(i + 1, j)
                        )
                        write_wav(sample_path, audio, 16000)
                        print(
                            "Saved real and fake samples into {}...".format(sample_path)
                        )
                        if j >= 6:
                            break

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
