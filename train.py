import os
import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import models
from datasets import ICONTRAIN 
from utils import AverageMeter
from IPython import embed

def main(args):
    model = models.get_model(load_pretrain=True).to(device=args.dev)
    train_set = ICONTRAIN(scale_size=56)
    train_loader = data.DataLoader(dataset=train_set,
            num_workers=args.num_workers, shuffle=True,
            batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)

        if epoch % args.save_freq == 0:
            if not os.path.exists(args.dump_dir):
                os.makedirs(args.dump_dir)
            filename = os.path.join(args.dump_dir, 
                    "fake_icon_res18_%03d.pth" % epoch)
            torch.save(model.state_dict(), filename)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model.train()

    end = time.time()
    log = ''
    for i, pack in enumerate(train_loader):
        input, target = pack
        data_time.update(time.time() - end)
        input = input.to(device=args.dev)
        target = target.to(device=args.dev)

        output = model(input)
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target)
        losses.update(loss.data, input.size(0))
        acc.update(prec1[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            log = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                   'Date {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Acc {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                    epoch, i+1, len(train_loader), batch_time=batch_time,
                        data_time = data_time, loss=losses, acc=acc 
                       ))
            print(log)

def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100 / batch_size)

def adjust_learning_rate(optimizer, epoch, args):
    interval = int(args.epochs * 0.4) 
    lr = args.lr * (0.1 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fake icon rec')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
            help='number of total epochs to run')
    parser.add_argument('--batch-size', type=int, default=64,
            help='number of traing batch size')
    parser.add_argument('--num-workers', type=int, default=4, 
            help='number of workers to process training data')
    parser.add_argument('--lr', type=float, default=1e-3,
            help='the base learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
            help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
            help='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int,
            help='print frequency')
    parser.add_argument('--save-freq', default=5, type=int, 
            help='save checkpoint frequency')
    parser.add_argument('--dev', default='cpu',  type=str,
            help='training device')
    parser.add_argument('--dump-dir', default='checkpoints',  type=str,
            help='training device')

    args = parser.parse_args()
    main(args)
