# 预备函数，用于模型加载等
from builtins import ImportError # type: ignore
import argparse
import torch
import numpy as np
from torchvision import datasets
from pytorch_image_classification import create_transform

try:
    import apex #type:ignore
except ImportError:
    pass
import numpy as np # type: ignore
import torch# type: ignore
import torch.distributed as dist# type: ignore
import torchvision.transforms as transforms



from pytorch_image_classification import (
    create_model,
    get_default_config,
    update_config,
)
from pytorch_image_classification.utils import (
    set_seed,
    setup_cudnn,
)
from pytorch_image_classification import (
    get_default_config,
    create_model,
    create_transform,
)

global global_step
global_step = 0

def load_config(configs_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default = configs_path) # type: ignore
    parser.add_argument('--local_rank', type=int, default=0) # type: ignore
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    config.merge_from_list(['train.dist.local_rank', args.local_rank])
    config = update_config(config)
    config.freeze()
    return config

def load_model_from_pth(configs_path,model_pth):
    '''
    configs_path: path of model config which contains model's defination, configurations and so on
    model_pth: path of train-finished model's pth file
    '''
    config = load_config(configs_path)
    set_seed(config)
    setup_cudnn(config)
    epoch_seeds = np.random.randint(np.iinfo(np.int32).max // 2,size=config.scheduler.epochs)
    if config.train.distributed:
        dist.init_process_group(backend=config.train.dist.backend,
                                init_method=config.train.dist.init_method,
                                rank=config.train.dist.node_rank,
                                world_size=config.train.dist.world_size)
        torch.cuda.set_device(config.train.dist.local_rank)
    model = create_model(config)
    checkpoint = torch.load(model_pth, weights_only=False)
    model.load_state_dict(checkpoint['model'],strict=False)
    device = torch.device("cpu")
    model.to(device)
    _ = model.eval()
    return model


def get_attack_loader(attack_set_path,configs_path,
        batch_size = 128,
        num_workers = 32,
        #img_size = 128,
        ):
    """
    attack_set_path：path of adversatial examples file path
    configs_path: as its name 
    """
    config = load_config(configs_path)
    attack_transform = create_attack_transform()
    dataset = datasets.ImageFolder(root= attack_set_path,transform=attack_transform)
    test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=None,
                shuffle=False,
                drop_last=False,
                pin_memory=True)
    return test_loader

def at_loader(attack_set_path,configs_path,
        batch_size = 128,
        num_workers = 32,
        ):
    """
    attack_set_path：path of adversatial examples file path
    configs_path: as its name 
    """
    config = load_config(configs_path)
    val_transform = create_transform(config, is_train=True)
    dataset = datasets.ImageFolder(root= attack_set_path,transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=None,
                shuffle=False,
                drop_last=False,
                pin_memory=True)
    return test_loader


def create_attack_transform():
    transform_list = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_list.append(transforms.Resize((128, 128)))
    transform_list.append(transforms.ToTensor())
    #transform_list.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(transform_list)
    return transform

