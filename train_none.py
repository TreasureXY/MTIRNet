from tqdm import tqdm
import torch
import torch.cuda
import os
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Grad_none
from data.div2k import DIV2K
from data.common import quantize
from option import parser
from common import adjust_lr, AverageMeter, calc_psnr, save_checkpoint

'''
python3.6 train_none.py --select_sigma all --gpu --gpu_id 0
'''
def train(dataset, loader, model, criterion, optimizer, device):
    losses = AverageMeter()

    model.train()

    with tqdm(total=len(dataset)) as t:
        t.set_description("train")

        for data in loader:
            lr, hr, sigma = data
            lr = lr.to(device)
            hr = hr.to(device)
            # sigma = sigma.to(device)

            sr = model(lr)
            loss = criterion(sr, hr)

            losses.update(loss.item(), lr.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss="{:.4f}".format(losses.avg))
            t.update(lr.shape[0])


def valid(dataset, loader, model, criterion, device):
    losses = AverageMeter()
    psnrs = AverageMeter()

    model.eval()

    with tqdm(total=len(dataset)) as t:
        t.set_description("valid")

        for data in loader:
            lr, hr, sigma = data
            lr = lr.to(device)
            hr = hr.to(device)

            with torch.no_grad():
                sr = model(lr)
                loss = criterion(sr, hr)
                losses.update(loss.item(), lr.shape[0])

                sr = quantize(sr, [0, 255])
                psnr = calc_psnr(sr, hr)
                psnrs.update(psnr.item(), lr.shape[0])

            t.set_postfix(loss='{:.4f}'.format(losses.avg))
            t.update(lr.shape[0])

        return psnrs.avg


def main():
    global args, model

    args = parser.parse_args()
    print(args)

    if args.gpu and not torch.cuda.is_available():
        raise Exception("No GPU found!")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    torch.manual_seed(2020)

    cudnn.benchmark = True
    device = torch.device(('cuda:' + args.gpu_id) if args.gpu else 'cpu')

    model = Grad_none.GRAD(feats=args.feats, basic_conv=args.basic_conv, tail_conv=args.tail_conv).to(device)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.L1Loss()

    if args.continue_train:
        checkpoint_file = torch.load(args.checkpoint_file)
        model.load_state_dict(checkpoint_file['model'])
        optimizer.load_state_dict(checkpoint_file['optimizer'])
        start_epoch = checkpoint_file['epoch']
        best_epoch = checkpoint_file['best_epoch']
        best_psnr = checkpoint_file['best_psnr']
        print("continue train {}.".format(start_epoch))
    else:
        start_epoch = 0
        best_epoch = 0
        best_psnr = 0

    print("Loading dataset ...")
    train_dataset = DIV2K(args, train=True)
    valid_dataset = DIV2K(args, train=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)

    checkpoint_name = "latest.pth"
    is_best = False
    for epoch in range(start_epoch + 1, args.epochs + 1):
        lr = adjust_lr(optimizer, args.lr, epoch, args.decay_step, args.decay_gamma)
        print("[epoch:{}/{}]".format(epoch, args.epochs))

        train(train_dataset, train_dataloader, model, criterion, optimizer, device)

        if epoch >= 90:
            valid_psnr = valid(valid_dataset, valid_dataloader, model, criterion, device)

            is_best = valid_psnr > best_psnr
            if is_best:
                best_psnr = valid_psnr
                best_epoch = epoch
            print("PSNR: {:.4f}".format(valid_psnr))

            print("learning rate: {}".format(lr))
            print("best PSNR: {:4f} in epoch: {}".format(best_psnr, best_epoch))

        save_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'best_epoch': best_epoch,
            }, os.path.join(args.checkpoint_dir, checkpoint_name), is_best)


if __name__ == '__main__':
    main()
