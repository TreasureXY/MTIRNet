import glob
import h5py
import torch
from tqdm import tqdm
from .common import load_img, img2np, get_patch, np2tensor, add_noise


class DIV2K(object):
    def __init__(self, args, train, n_pairs=None):
        self.select_sigma = args.select_sigma
        self.dataset_dir = args.dataset_dir
        self.h5file_dir = args.h5file_dir
        self.batch_size = args.batch_size
        self.iterations_per_epoch = args.iterations_per_epoch
        self.patch_size = args.patch_size
        self.augment_patch = args.augment_patch
        self.feats = args.feats
        self.train = train

        if self.select_sigma == "all":
            self.train_dir_h = self.dataset_dir + "/training_H_mix/"
            self.valid_dir_h = self.dataset_dir + "/validation_H_mix/"
        else:
            self.train_dir_h = self.dataset_dir + "/training_H/"
            self.valid_dir_h = self.dataset_dir + "/validation_H/"

        self.h_dir = {"train": self.train_dir_h, "valid": self.valid_dir_h}

        if not n_pairs:
            self.n_pairs = 800 if self.train else 100
        else:
            self.n_pairs = n_pairs

        if not self._is_ready():
            self._prepare()

    def _is_ready(self):
        try:
            for phase, num_images in [("train", 800), ("valid", 100)] if self.train else [("valid", 100)]:
                with h5py.File("{}/DIV2K_np_{}_{}.h5".format(self.h5file_dir, phase, self.select_sigma), 'r') as h5:
                    assert len(h5['h']) == num_images and len(h5['l']) == num_images and len(h5['sigma']) == num_images
        except Exception:
            return False
        return True

    def _prepare(self):
        print("Preparing DIV2K dataset ...")
        for phase in ["train", "valid"] if self.train else ["valid"]:
            print(phase)
            h5file_path = "{}/DIV2K_np_{}_{}.h5".format(self.h5file_dir, phase, self.select_sigma)
            h5 = h5py.File(h5file_path, 'w')
            h_group = h5.create_group('h')
            l_group = h5.create_group('l')
            sigma_group = h5.create_group('sigma')

            h_list = sorted(glob.glob(self.h_dir[phase] + "*.png"))

            if self.select_sigma == "all":
                sigma_list = [float(h_path[-6:-4]) for h_path in h_list]
            else:
                sigma_list = [float(self.select_sigma)] * len(h_list)

            with tqdm(total=len(h_list)) as t:
                t.set_description("H & L")
                for i, path in enumerate(h_list):
                    img = img2np(load_img(path))
                    h_group.create_dataset(str(i), data=img)
                    l_group.create_dataset(str(i), data=add_noise(img, sigma_list[i], self.train))
                    t.update()

            with tqdm(total=len(sigma_list)) as t:
                t.set_description("sigma")
                for i, sigma in enumerate(sigma_list):
                    sigma_group.create_dataset(str(i), data=sigma)
                    t.update()

            h5.close()
        print("Prepare successfully")

    def __len__(self):
        if self.train:
            return self.batch_size * self.iterations_per_epoch
        else:
            return self.n_pairs

    def __getitem__(self, idx):
        idx = str(idx % self.n_pairs)

        with h5py.File(
                "{}/DIV2K_np_{}_{}.h5".format(self.h5file_dir, 'train' if self.train else 'valid', self.select_sigma),
                'r') as h5:
            if self.train:
                lr, hr = get_patch(h5['l'][idx], h5['h'][idx], self.patch_size, self.augment_patch)
                lr = np2tensor(lr)
                hr = np2tensor(hr)

                sigma = h5['sigma'][idx][()]
                sigma = torch.tensor([float(sigma)])
                sigma = sigma.reshape(1 * 1 * 1).expand(self.feats, self.patch_size, self.patch_size)
            else:
                lr, hr = h5['l'][idx][()], h5['h'][idx][()]
                lr = np2tensor(lr)
                hr = np2tensor(hr)

                sigma = h5['sigma'][idx][()]
                sigma = torch.tensor([float(sigma)])
                expand_H = lr.shape[1]
                expand_W = lr.shape[2]
                sigma = sigma.reshape(1 * 1 * 1).expand(self.feats, expand_H, expand_W)

        h5.close()
        return lr, hr, sigma
