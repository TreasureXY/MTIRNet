import argparse

parser = argparse.ArgumentParser(description="Grad")

# data set
# 噪声水平的值，取10，20，30，40，all
parser.add_argument("--select_sigma", type=str, default="10", help="noise level")
parser.add_argument("--dataset_dir", type=str, default="../MyDIV2K", help="root of the data set")
parser.add_argument("--h5file_dir", type=str, default="../MyDIV2K")
parser.add_argument("--patch_size", type=int, default=128)
parser.add_argument("--augment_patch", type=bool, default=True)

# train
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--iterations_per_epoch", type=int, default=1000)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--decay_step", type=int, default=50)
parser.add_argument("--decay_gamma", type=float, default=0.5)

# continue train
parser.add_argument("--continue_train", action="store_true")
parser.add_argument("--checkpoint_file", type=str, default="./checkpoint/latest.pth")

# network
parser.add_argument("--feats", type=int, default=64)
parser.add_argument("--basic_conv", type=int, default=1)
parser.add_argument("--tail_conv", type=int, default=2)


# test any dataset
parser.add_argument("--test_h", type=str, default="../MyDIV2K/validation_H/")
parser.add_argument("--test_sigma", type=str, default="10")
parser.add_argument("--test_checkpoint", type=str, default="./checkpoint/best.pth")
parser.add_argument("--test_result", type=str, default="./test_result")
parser.add_argument("--test_save", action="store_true")


