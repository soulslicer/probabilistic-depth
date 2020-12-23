import os
import json
import pprint
import datetime
import argparse
from path import Path
from easydict import EasyDict

from logger import init_logger

import torch
from utils.torch_utils import init_seed
import torch.multiprocessing as mp
import torch.distributed as dist

from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer

def worker(id, args): pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/default.json')
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--viz', action='store_true', help='viz', default=False)
    parser.add_argument('--eval', action='store_true', help='viz', default=False)
    parser.add_argument('--resume', action='store_true', help='viz', default=False)
    parser.add_argument('--init_model', default='')
    parser.add_argument('--write_video', default='')

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
        config["write_video"] = args.write_video
        cfg = EasyDict(config)

    if args.viz:
        cfg.var.viz = True

    setattr(cfg.train, 'init_model', args.init_model)

    init_seed(cfg.seed)

    # Overwrite
    cfg.train.batch_size = args.batch_size

    # store files day by day
    exp_name = cfg.data.exp_name
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    cfg.save_root = Path('./outputs/checkpoints/') + exp_name
    cfg.save_root.makedirs_p()

    # Resume
    if args.resume:
        model_path = Path('./outputs/checkpoints/') + exp_name + "/" + exp_name + "_ckpt.pth.tar"
        if not os.path.isfile(model_path):
            print("Unable to find model at default location")
        else:
            cfg.train.pretrained_model = model_path

    # Eval Mode
    if args.eval:
        model_path = Path('./outputs/checkpoints/') + exp_name + "/" + exp_name + "_model_best.pth.tar"
        if not os.path.isfile(model_path):
            raise RuntimeError("Unable to find model at default location")
        cfg.train.pretrained_model = model_path
        cfg.var.mload = False
        cfg.mp.enabled = False
        cfg.train.n_gpu = 1
        cfg.train.batch_size = 1
        cfg.eval = True

    # Multiprocessing (Number of Workers = Number of GPU)
    if cfg.mp.enabled:
        # Checks
        if cfg.train.n_gpu > 0:
            if cfg.train.n_gpu > torch.cuda.device_count():
                cfg.train.n_gpu = torch.cuda.device_count()
                print("Total GPU size is incorrect. Lowering to Base")
            cfg.mp.workers = cfg.train.n_gpu
        else:
            if cfg.mp.workers <= 0:
                raise RuntimeError("Wrong number of workers")

        # Set Flags
        os.environ["MASTER_ADDR"] = cfg.mp.master_addr
        os.environ["MASTER_PORT"] = str(cfg.mp.master_port)
        os.environ["WORLD_SIZE"] = str(cfg.mp.workers)
        os.environ["RANK"] = str(0)
        shared = torch.zeros((cfg.mp.workers, 10)).share_memory_()

        # Spawn Worker
        mp.spawn(worker, nprocs=cfg.mp.workers, args=(cfg, shared))
    else:
        # Spawn Worker
        shared = torch.zeros((1, 10)).share_memory_()
        worker(0, cfg, shared)


def worker(id, cfg, shared):
    # init logger
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    _log = init_logger(log_dir=cfg.save_root, filename=curr_time[6:] + '.log')
    if id == 0: _log.info(id, '=> will save everything to {}'.format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    if id == 0: _log.info(id, '=> configurations \n ' + cfg_str)

    # Distributed
    if cfg.mp.enabled:
        if cfg.train.n_gpu > 0:
            dist.init_process_group(backend="nccl", init_method="env://",
                                    world_size=cfg.mp.workers, rank=id)
        else:
            dist.init_process_group(backend="gloo", init_method="env://",
                                    world_size=cfg.mp.workers, rank=id)

    # Get Model and Loss
    model = get_model(cfg, id)
    loss = get_loss(cfg, id)

    # Create Trainer
    trainer = get_trainer(cfg)(id, model, loss, _log, cfg.save_root, cfg, shared)

    # Train or Test
    try:
        if cfg.eval:
            trainer.eval()
        else:
            trainer.train()
    except Exception as e:
        import traceback
        traceback.print_exc()

    # Destroy
    if cfg.mp.enabled:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()