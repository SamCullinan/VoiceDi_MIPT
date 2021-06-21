import argparse
from solver_stargan import Solver
from torch.backends import cudnn
import datetime
import os
import sys

sys.path.insert(1, "/vol/bitbucket/apg416/project/decomposition")
from iemocap_loader import get_loader


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
    loader = get_loader(
        config.selected_emos, config.mode, config.batch_size, config.num_workers
    )

    # Solver for training and testing StarGAN.
    solver = Solver(loader, config)

    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    timestamp = str(datetime.datetime.now().strftime("%d_%m_%H_%M"))

    # Model configuration.
    parser.add_argument(
        "--c_dim", type=int, default=4, help="dimension of domain labels"
    )
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=10,
        help="weight for domain classification loss",
    )
    parser.add_argument(
        "--lambda_rec", type=float, default=10, help="weight for reconstruction loss"
    )
    parser.add_argument(
        "--lambda_gp", type=float, default=10, help="weight for gradient penalty"
    )
    parser.add_argument(
        "--lambda_id", type=float, default=10, help="weight for identity loss"
    )

    # Training configuration.
    parser.add_argument("--batch_size", type=int, default=16, help="mini-batch size")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=2000000,
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

    # Test configuration.
    parser.add_argument(
        "--test_iters", type=int, default=200000, help="test model from this step"
    )

    # Miscellaneous.
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--use_tensorboard", type=str2bool, default=True)

    # Directories.
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/codegan/experiments/logs",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/codegan/experiments/models",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/codegan/experiments/samples",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/vol/bitbucket/apg416/project/codegan/experiments/results",
    )

    # Step size.
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=25000)
    parser.add_argument("--model_save_step", type=int, default=25000)
    parser.add_argument("--lr_update_step", type=int, default=1000)

    parser.add_argument(
        "--selected_emos",
        "--list",
        nargs="+",
        help="selected emotions from IEMOCAP dataset",
        default=["ang", "hap", "sad", "neu"],
    )
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--code", type=str, default="content")

    config = parser.parse_args()
    print(config)
    main(config)
