# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import wandb

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

from models.dino import utils
from models.dino import vision_transformer as vits

# load bigearthnet dataset
from datasets.BigEarthNet.bigearthnet_dataset_seco_lmdb_s2_uint8 import LMDBDataset, random_subset, LMDBDatasetRA
from cvtorchvision import cvtransforms
### end of change ###
import pdb

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score
import builtins
import sys

def eval_linear(args):
    utils.init_distributed_mode(args)
    if args.rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    print({k: getattr(args, k) for k in dir(args) if not k.startswith('_')})

    in_channels = 13
    # in_channels = 12

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, in_chans=in_channels)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Identity()
        #model.fc = torch.nn.Linear(2048,19)
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=19)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============

    train_transform = cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(224, scale=(0.8,1.0)),
        # cvtransforms.RandomResizedCrop(224),
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.RandomVerticalFlip(),
        cvtransforms.ToTensor(),
        #cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    #dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)

    val_transform = cvtransforms.Compose([
        cvtransforms.Resize(256),
        cvtransforms.CenterCrop(224),
        cvtransforms.ToTensor(),
        #cvtransforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])



    lmdb_train = 'train_B12.lmdb'
    lmdb_val = 'val_B12.lmdb'
    if args.bands == 'RGB':
        args.n_channels = 3
    elif args.bands == 'B12':   
        args.n_channels = 12
    elif args.bands == 'B13' or args.bands == 'all':
        args.n_channels = 13
    elif args.bands == 'B2':
        args.n_channels = 2
    elif args.bands == 'B14':
        args.n_channels = 14


    dataset_train = LMDBDatasetRA(
        set_type='train',
        transform=train_transform,
        is_slurm_job=args.is_slurm_job,
        file_index=args.file_index
    )
    dataset_val = LMDBDatasetRA(
        set_type='val',
        transform=val_transform,
        is_slurm_job=args.is_slurm_job,
        file_index=args.file_index
    )        

    if args.train_frac is not None and args.train_frac<1:
        dataset_train = random_subset(dataset_train,args.train_frac,seed=args.seed)


    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    

    #dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return    
    
            
    
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    if args.resume:
        utils.restart_from_checkpoint(
            os.path.join(args.checkpoints_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    
    if args.rank==0 and not os.path.isdir(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir,exist_ok=True)
    
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            wandb.log({"val acc1": test_stats["acc1"], "epoch": epoch})
        if utils.is_main_process():
            with (Path(args.checkpoints_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.checkpoints_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (images, target) in metric_logger.log_every(loader, 20, header):

        b_zeros = torch.zeros((images.shape[0],1,images.shape[2],images.shape[3]),dtype=torch.float32)
        inp = torch.cat((images[:,:10,:,:],b_zeros,images[:,10:,:,:]),dim=1)

        # inp = images
        
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.MultiLabelSoftMarginLoss()(output, target.long())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        wandb.log({"train loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for images, target in metric_logger.log_every(val_loader, 20, header):

        b_zeros = torch.zeros((images.shape[0],1,images.shape[2],images.shape[3]),dtype=torch.float32)
        inp = torch.cat((images[:,:10,:,:],b_zeros,images[:,10:,:,:]),dim=1)
        # inp = images

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.MultiLabelSoftMarginLoss()(output, target.long())

        '''
        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))
        '''
        score = torch.sigmoid(output).detach().cpu()
        acc1 = average_precision_score(target.cpu(), score, average='micro') * 100.0
        acc5 = acc1
        
        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class FakeArgs:
    arch = 'vit_small'
    avgpool_patchtokens = False
    bands = 'all'
    batch_size_per_gpu = 64
    checkpoint_key = 'teacher'
    checkpoints_dir = '/var/node433/local/ryan_a/new_data/ben_checkpoints/distillation_l2_normalised/'
    data_path = ''
    dist_url = 'env://'
    epochs = 100
    evaluate = False
    file_index = 2
    gpu = 0
    is_slurm_job = True
    lr = 0.1
    n_last_blocks = 4
    num_workers = 16
    patch_size = 16
    pretrained = '/var/node433/local/ryan_a/data/ssl4eo_ssl/ssl4eo_ssl/distillation_l2_normalised/checkpoint.pth'
    resume = True
    seed = 42
    val_freq = 5
    train_frac = 0.1

if __name__ == '__main__':
    # parser = argparse.ArgumentParser('Evaluation with linear classification on BigEarthNet.')
    # parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
    #     for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    # parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
    #     help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
    #     We typically set this to False for ViT-Small and to True with ViT-Base.""")
    # parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    # parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained weights to evaluate.")
    # parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    # parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    # parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
    #     training (highest LR used during training). The learning rate is linearly scaled
    #     with the batch size, and specified here for a reference batch size of 256.
    #     We recommend tweaking the LR depending on the checkpoint evaluated.""")
    # parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    # parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    #     distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    # parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    # parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    # parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    # parser.add_argument('--val_freq', default=5, type=int, help="Epoch frequency for validation.")
    # parser.add_argument('--checkpoints_dir', default=".", help='Path to save logs and checkpoints')
    # parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    # parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    #
    # parser.add_argument('--lmdb_dir', default='/path/to/imagenet/', type=str, help='Please specify path to the ImageNet folder.')
    # parser.add_argument('--bands', type=str, default='all', help="input bands")
    # parser.add_argument("--lmdb", action='store_true', help="use lmdb dataset")
    # parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")
    # parser.add_argument("--resume", action='store_true', help="resume from checkpoint")
    # parser.add_argument("--train_frac", default=1.0, type=float, help="use a subset of labeled data")
    # parser.add_argument("--seed",default=42,type=int)
    #
    # args = parser.parse_args()

    # [print([k, getattr(args, k)]) for k in dir(args) if not k.startswith('_')]
    #
    # raise ValueError()
    #
    #
    args = FakeArgs()

    run = wandb.init(
        # Set the project where this run will be logged
        project="MarineDebrisSSL",
        name='linear_BE_dino_original_myweights',
        # Track hyperparameters and run metadata
        config={k: getattr(args, k) for k in dir(args) if not k.startswith('_')},
    )

    
    eval_linear(args)
