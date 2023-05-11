import os
import math
import time
import shutil
import argparse

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import data
import models
from utils.logger import Logger
from utils.scheduler import get_lr
from utils.env import get_options, prepare_env


class Trainer:
    def __init__(self, args, local_rank, training=True):
        self.args = args
        self.model_without_ddp = getattr(models, args.model)(args).to(args.device)

        if local_rank != -1:
            self.model = DDP(self.model_without_ddp, device_ids=[local_rank], output_device=local_rank)
        else:
            self.model = self.model_without_ddp

        if training:
            # Do not train text encoder
            pretrained_backbone_weight = []
            added_backbone_weight = []
            img_encoder_added_weight = self.model_without_ddp.img_backbone.added_weight_names
            for name, p in self.model_without_ddp.img_backbone.named_parameters():
                if name not in img_encoder_added_weight:
                    pretrained_backbone_weight.append(p)
                else:
                    added_backbone_weight.append(p)
            self.optimizer = optim.AdamW([
                {"params": pretrained_backbone_weight, "lr": args.backbone_lr},
                {"params": added_backbone_weight},
                {"params": self.model_without_ddp.enhancer.parameters()},
                {"params": self.model_without_ddp.decoder.parameters()}
            ], lr=args.start_lr, weight_decay=args.weight_decay)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def device(self):
        self.model.to(self.args.device)

    def inference(self, x):
        return self.model_without_ddp.inference(x)

    def train_one_step(self, inp_dict, lr, set_img_dict=False):
        # with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
        for param_group in self.optimizer.param_groups[1:]:
            param_group['lr'] = lr

        total_loss, log_dict = self.model(inp_dict, set_img_dict=set_img_dict)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return log_dict

    def save_model(self, path, epoch, step, best, save_optim=False):
        chkpt = {
            'model': self.model_without_ddp.state_dict(),
            'best_performace': best,
            'step': step,
            'epoch': epoch,
        }
        if save_optim:
            chkpt.update({
                'optimizer': self.optimizer.state_dict(),
            })
        torch.save(chkpt, path)


def train(args, trainer):
    local_rank = args.local_rank
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(args.log_dir, 'config.yaml'))
        summary_writer = SummaryWriter(args.log_dir)
        logger = Logger(summary_writer, metric_summary_freq=args.metric_summary_freq)

        # Print Configuration
        print(args)

        # Print Model Information
        print(trainer.model)
        num_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print('Number of params:', num_params)

        # Prepare training
        step = 0
        epoch = 0
        best_mae = math.inf
        # Todo: Implement resume

        train_dataset = getattr(data, f'{args.data_name}')(args, mode='train')
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True, sampler=train_sampler)

        time_stamp = time.time()
        for cur_epoch in range(epoch, args.total_epochs):
            train_sampler.set_epoch(cur_epoch)

            trainer.train()
            for batch in train_dataloader:
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()

                cur_lr = get_lr(args.start_lr, args.end_lr, args.warm_up_iter, step + 1, args.last_lr_decay_iter)

                # Update model weight
                set_img_dict = local_rank == 0 and (step + 1) % args.img_summary_freq == 0
                metrics = trainer.train_one_step(batch, cur_lr, set_img_dict)

                train_time_interval = time.time() - time_stamp

                step += 1

                if local_rank == 0:
                    metrics.update({
                        'lr': cur_lr,
                        'train_time': train_time_interval,
                        'data_time': data_time_interval,
                    })
                    logger.push(metrics)

                    # Image logging
                    if set_img_dict:
                        img_dict = trainer.model_without_ddp.get_log_dict()
                        logger.add_image_summary(img_dict)

                    # Save model weighs frequently with optimizer
                    if step % args.save_latest_freq == 0:
                        path = os.path.join(args.log_dir, 'latest.pth')
                        trainer.save_model(path, cur_epoch, step, best_mae, save_optim=True)

            if local_rank == 0:
                print(f"Training epoch {cur_epoch} Done")
                if (cur_epoch + 1) % args.valid_freq_epoch == 0:
                    val_results, img_dict = trainer.validate_fsc384(set_img_dict=True)

                    cur_mae = val_results['val/fsc_mae']
                    if cur_mae < best_mae:
                        best_mae = cur_mae
                        path = os.path.join(args.log_dir, f'best_mae.pth')
                        trainer.save_model(path, (cur_epoch + 1), step, best_mae, save_optim=False)

                    logger.write_dict(val_results, step=cur_epoch + 1)
                    logger.add_image_summary(img_dict, step=cur_epoch + 1)
                    print(f"Epoch {cur_epoch} Validation Done - Best: {best_mae:.3f}")

                if (cur_epoch + 1) % args.save_every_freq_epoch == 0:
                    path = os.path.join(args.log_dir, f'epoch_{cur_epoch + 1:03d}.pth')
                    trainer.save_model(path, cur_epoch + 1, step, best_mae, save_optim=True)

            dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EuiyeonKim Zeroshot Object Counter')
    parser.add_argument('--config', type=str, default='configs/debug.yaml')
    parser.add_argument('--exp_name', default='debug', type=str)

    args = get_options(parser.parse_args())
    prepare_env(args)

    trainer = Trainer(args, args.local_rank)

    train(args, trainer)

    dist.destroy_process_group()

