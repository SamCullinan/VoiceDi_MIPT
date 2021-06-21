import os
import argparse
from solver_wgan import Solver
from torch.backends import cudnn
import datetime
from torch.utils import data
import torch
import numpy as np
import pandas as pd


class EmbeddingsDataset(data.Dataset):
    def __init__(self, data_dir):
        self.meta = pd.read_csv(os.path.join(data_dir, "meta_cv.csv"))

    def __getitem__(self, idx):
        path, y = self.meta.loc[idx, :]
        x = torch.from_numpy(np.load(path))
        return x, y

    def __len__(self):
        return len(self.meta)


def str2bool(v):
    return v.lower() in ("true")


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    loader = data.DataLoader(
        EmbeddingsDataset(config.data_dir),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # Solver for training and testing StarGAN.
    solver = Solver(loader, config)

    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    timestamp = str(datetime.datetime.now().strftime("%d_%m_%H_%M"))

    # Model configuration.
    parser.add_argument(
        "--c_dim", type=int, default=3, help="dimension of domain labels"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="weight for gradient penalty"
    )
    parser.add_argument(
        "--lambda_cls", type=float, default=1, help="weight for classification loss"
    )

    # Training configuration.
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=1000000,
        help="number of total iterations for training D",
    )
    parser.add_argument(
        "--num_iters_decay",
        type=int,
        default=1900000,
        help="number of iterations for decaying lr",
    )
    parser.add_argument(
        "--g_lr", type=float, default=0.0001, help="learning rate for G"
    )
    parser.add_argument(
        "--d_lr", type=float, default=0.0001, help="learning rate for D"
    )
    parser.add_argument(
        "--n_critic", type=int, default=5, help="number of D updates per each G update"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--resume_iters", type=int, default=None, help="resume training from this step"
    )

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--use_tensorboard", type=str2bool, default=True)

    # Directories.
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/cvoicegan/experiments/logs",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/cvoicegan/experiments/models",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/cvoicegan/experiments/samples",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/cvoicegan/experiments/results",
    )

    # Step size.
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=50000)
    parser.add_argument("--model_save_step", type=int, default=50000)
    parser.add_argument("--lr_update_step", type=int, default=1000)

    parser.add_argument(
        "--data_dir", type=str, default="/vol/bitbucket/apg416/embs_covo/"
    )
    parser.add_argument("--dim_z", type=int, default=256)
    parser.add_argument("--dim_h", type=int, default=1024)

    config = parser.parse_args()
    print(config)
    main(config)
