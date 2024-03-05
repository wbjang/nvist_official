
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import math
import torch
from torch import nn, Tensor
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.transforms import ToTensor
import numpy as np
import cv2
from PIL.Image import fromarray
from collections import OrderedDict

def load_pretrained_weights(ckpt: Dict[str, Any], model: nn.Module) -> nn.Module:
    new_state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def load_pretrained_when_shape_matches(model: nn.Module, model_pretrained: Dict[str, Tensor], show_unloaded_shapes: bool=False) -> nn.Module:
    names = []
    for name, param in model_pretrained.named_parameters():
        if name in model.state_dict():
            if param.shape == model.state_dict()[name].shape:
                model.state_dict()[name].copy_(param)
            else: 
                names.append(name)
        else: 
            names.append(name)
    print("we loaded the pretrained weights")
    if show_unloaded_shapes: 
        print("except at " + str(names))
    return model

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def visualize_images(imgs: Tensor, nrow: Optional[int]=None) -> Tensor: # nchw -> hwc
    if nrow is None: 
        grid_img = make_grid(imgs.detach().cpu())
    else: 
        grid_img = make_grid(imgs.detach().cpu(), nrow=nrow)
    grid_img *= 255
    grid_img = grid_img.to(torch.uint8)
    return grid_img.permute(1,2,0) # hwc

def put_gt_output_side_by_side(gt_images: Tensor, output_images: Tensor) -> Tensor:
    output = [torch.cat([gt_image.detach().cpu(), output_image.detach().cpu()], dim=-1).unsqueeze(0) for gt_image, output_image in zip(gt_images, output_images)]
    output = torch.cat(output)
    return visualize_images(output)

def visualize_input_novel_view_gt(input_images: Tensor, output_images: Tensor, gt_images: Tensor, depth_images: Tensor) -> Tensor:
    input_images, output_images, gt_images, depth_images = input_images.detach().cpu(), output_images.detach().cpu(), gt_images.detach().cpu(), depth_images.detach().cpu()
    output = [torch.cat([input_image, output_image, gt_image, depth_image], dim=-1).unsqueeze(0) for input_image, output_image, gt_image, depth_image in zip(input_images, output_images, gt_images, depth_images)]
    return visualize_images(torch.cat(output), nrow=8)

    
def adjust_learning_rate(current_iters: int, warmup_iters: int, total_iters: int, lr_original: float, lr_min: float, optimizer: torch.optim.Optimizer) -> float:

    if current_iters < warmup_iters:
        lr = lr_original * current_iters / warmup_iters 
    else:
        """Decay the learning rate with half-cycle cosine after warmup"""
        lr = lr_min + (lr_original - lr_min) * 0.5 * \
            (1. + math.cos(math.pi * ((current_iters - warmup_iters) / (total_iters-warmup_iters))))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def cycle(dl: torch.utils.data.DataLoader) -> Dict:
    while True:
        for data in dl:
            yield data

def unnormalize_imgs_imagenet(imgs: Tensor) -> Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    if len(imgs.shape) == 3:
        return imgs * std.reshape(1,1,3).type_as(imgs) + mean.reshape(1,1,3).type_as(imgs)
    elif len(imgs.shape) == 4 and imgs.shape[1] == 3:
        return imgs * std.reshape(1,3,1,1).type_as(imgs) + mean.reshape(1,3,1,1).type_as(imgs)
    elif len(imgs.shape) == 4 and imgs.shape[-1] == 3:
        return imgs * std.reshape(1,1,1,3).type_as(imgs) + mean.reshape(1,1,1,3).type_as(imgs)
    
def normalize_imgs_imagenet(imgs: Tensor) -> Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    if len(imgs.shape) == 4 and imgs.shape[1] == 3:
        t = transforms.Normalize(mean, std)
        return t(imgs)
    else:
        return (imgs - mean.reshape(1,1,3).type_as(imgs)) / std.reshape(1,1,3).type_as(imgs)

def normalize_imgs(imgs: Tensor) -> Tensor:
    mean = torch.tensor([0.5,0.5,0.5])
    std = torch.tensor([0.5,0.5,0.5])
    if len(imgs.shape) == 4 and imgs.shape[1] == 3:
        mean = torch.tensor([0.5,0.5,0.5])
        std = torch.tensor([0.5,0.5,0.5])
        t = transforms.Normalize(mean, std)
        return t(imgs)
    else:
        return (imgs - mean.reshape(1,1,3).type_as(imgs)) / std.reshape(1,1,3).type_as(imgs)


def unnormalize_imgs(imgs: Tensor) -> Tensor:
    mean = torch.tensor([0.5,0.5,0.5])
    std = torch.tensor([0.5,0.5,0.5])
    if len(imgs.shape) == 3:
        return imgs * std.reshape(1,1,3).type_as(imgs) + mean.reshape(1,1,3).type_as(imgs)
    elif len(imgs.shape) == 4 and imgs.shape[1] == 3:
        return imgs * std.reshape(1,3,1,1).type_as(imgs) + mean.reshape(1,3,1,1).type_as(imgs)



def visualize_mae(model: nn.Module, pred_mask: Tensor, mask: Tensor, train_images: Tensor) -> Tensor:

    if isinstance(model, torch.nn.parallel.DistributedDataParallel): 
        pred_masked = model.module.unpatchify(pred_mask)
    else: 
        pred_masked = model.unpatchify(pred_mask)
    
    pred_masked = unnormalize_imgs_imagenet(pred_masked)
    pred_masked = pred_masked.clamp(0.,1.)
    mask = mask.unsqueeze(-1).repeat(1,1,pred_mask.shape[-1])
    if isinstance(model, torch.nn.parallel.DistributedDataParallel): 
        mask = model.module.unpatchify(mask)
    else: 
        mask = model.unpatchify(mask)

    images_filled = train_images.type_as(pred_masked) * (1-mask) + pred_masked * mask

    return images_filled


def visualize_depths(depths: Union[Tensor, np.ndarray], minmax:  Optional[Tuple[float, float]] = None) -> Tensor:
    depth_imgs = []
    for depth in depths:
        depth_imgs.append(visualize_depth(depth))
    return torch.cat(depth_imgs)

def visualize_depth(depth: Union[Tensor, np.ndarray], minmax: Optional[Tuple[float, float]] = None, cmap: int = cv2.COLORMAP_JET) -> Tensor:
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        if (x[x>0]).size > 0:
            mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        else: 
            mi = 0
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = fromarray(cv2.applyColorMap(x, cmap))
    x_ = ToTensor()(x_)  # (3, H, W)
    return x_.unsqueeze(0)



def turn_int_into_list(number: int) -> List[int]:
    return [number, number]

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self) -> Tensor:
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]
