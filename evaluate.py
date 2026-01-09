#!/usr/bin/env python

import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm # type: ignore

from fvcore.common.checkpoint import Checkpointer # type: ignore

from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_dataloader,
    create_loss,
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
)


def load_config():
    #the directory should be str type
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT/"configs/mstar/densenet.yaml") ,type=str, required=False)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.test.output_dir = str(ROOT/"experiments/mstar/densenet/test2-2")
    config.test.checkpoint = str(ROOT/"experiments/mstar/densenet/exp2-2/checkpoint_00500.pth") # type: ignore

    config.freeze()
    return config


def evaluate(config, model, test_loader, loss_func, logger):
    device = torch.device(config.device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)

            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())

            _, preds = torch.max(outputs, dim=1)
            pred_label_all.append(preds.cpu().numpy())

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)
    labels = np.concatenate(pred_label_all)
    return preds, probs, labels, loss_meter.avg, accuracy


def main():
    config = load_config()

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    checkpointer = Checkpointer(model,
                                save_dir=output_dir,
                                )
    checkpointer.load(config.test.checkpoint)

    test_loader = create_dataloader(config, is_train=False)
    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc = evaluate(config, model, test_loader,
                                               test_loss, logger)

    output_path = output_dir / f'predictions.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc)


if __name__ == '__main__':
    main()
