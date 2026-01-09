import importlib
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import yacs.config


def create_model(config: yacs.config.CfgNode) -> nn.Module:
    module = importlib.import_module(
        'pytorch_image_classification.models'
        f'.{config.model.type}.{config.model.name}')
    model = getattr(module, 'Network')(config)
    device = torch.device(config.device)
    model.to(device)
    return model




def apply_data_parallel_wrapper(config: yacs.config.CfgNode, model: nn.Module) -> nn.Module:
    if config.train.distributed and dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 使用 torchrun 提供的 LOCAL_RANK
        torch.cuda.set_device(local_rank)                   # 每个进程绑定对应 GPU
        model = model.to(local_rank)
        if config.train.dist.use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # 根据模型情况选择 True/False
        )
    else:
        model.to(config.device)
    return model



# def apply_data_parallel_wrapper(config: yacs.config.CfgNode,
#                                 model: nn.Module) -> nn.Module:
#     local_rank = config.train.dist.local_rank
#     if dist.is_available() and dist.is_initialized():
#         if config.train.dist.use_sync_bn:
#             model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#         model = nn.parallel.DistributedDataParallel(model,
#                                                     device_ids=[local_rank],
#                                                     output_device=local_rank)
#     else:
#         model.to(config.device)
#     return model
