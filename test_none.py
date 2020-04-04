from tqdm import tqdm
import torch
import glob
import torch.cuda
import os
import h5py
import torch.backends.cudnn as cudnn
import PIL.Image as pil_image
import Grad_none
from option import parser
from common import AverageMeter, calc_psnr
from data.common import load_img, np2tensor, img2np, quantize, add_noise

'''
python3.6 test_none.py --test_save --test_sigma 10 --test_result ./result_10 --gpu --gpu_id 0
python3.6 testx.py --test_sigma 10 --gpu --gpu_id 0 --test_save --test_l ../MyDIV2K/validation_L/30 --test_result ./test_result_30
'''


def is_ready(args):
    try:
        with h5py.File("{}/DIV2K_np_test_{}.h5".format(args.h5file_dir, args.test_sigma), 'r') as h5:
            assert len(h5['h']) == 100 and len(h5['l']) == 100
    except Exception:
        return False
    return True


def prepare_data(args):
    print("Preparing DIV2K dataset ...")
    h5file_path = "{}/DIV2K_np_test_{}.h5".format(args.h5file_dir, args.test_sigma)
    h5 = h5py.File(h5file_path, 'w')
    h_group = h5.create_group('h')
    l_group = h5.create_group('l')

    h_list = sorted(glob.glob(args.test_h + "*.png"))

    with tqdm(total=len(h_list)) as t:
        t.set_description("H & L")
        for i, path in enumerate(h_list):
            img = img2np(load_img(path))
            h_group.create_dataset(str(i), data=img)
            l_group.create_dataset(str(i), data=add_noise(img, float(args.test_sigma), train=False))
            t.update()

    h5.close()
    print("Prepare successfully")


def main():
    global args, model

    args = parser.parse_args()
    print(args)

    if args.gpu and not torch.cuda.is_available():
        raise Exception("No GPU found!")

    if not os.path.exists(args.test_result):
        os.makedirs(args.test_result)

    if not is_ready(args):
        prepare_data(args)

    cudnn.benchmark = True
    device = torch.device(('cuda:' + args.gpu_id) if args.gpu else 'cpu')

    model = Grad_none.GRAD(feats=args.feats, basic_conv=args.basic_conv, tail_conv=args.tail_conv)
    checkpoint_file = torch.load(args.test_checkpoint)
    model.load_state_dict(checkpoint_file['model'])
    model.eval()
    model = model.to(device)

    psnrs = AverageMeter()

    with tqdm(total=100) as t:
        t.set_description("test")

        for idx in range(0, 100):
            with h5py.File(
                    "{}/DIV2K_np_test_{}.h5".format(args.h5file_dir, args.test_sigma),
                    'r') as h5:
                l_image, h_image = h5['l'][str(idx)][()], h5['h'][str(idx)][()]
                l_image = np2tensor(l_image)
                h_image = np2tensor(h_image)

                l_image = l_image.unsqueeze(0)
                h_image = h_image.unsqueeze(0)

                l_image = l_image.to(device)
                h_image = h_image.to(device)

                with torch.no_grad():
                    output = model(l_image)
                    output = quantize(output, [0, 255])
                    psnr = calc_psnr(output, h_image)
                    psnrs.update(psnr.item(), 1)

                if args.test_save:
                    save_image_path = "{}/{:04d}.png".format(args.test_result, idx)
                    output = output.squeeze(0)
                    output = output.data.permute(1, 2, 0)
                    save_image = pil_image.fromarray(output.byte().cpu().numpy())
                    save_image.save(save_image_path)

            t.update(1)

    print("PSNR: {:.4f}".format(psnrs.avg))


if __name__ == '__main__':
    main()
