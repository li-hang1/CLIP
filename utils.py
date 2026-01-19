import torch
import torch.distributed as dist
import yaml
import json

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    tensor: [B, embed_dim]
    return: [world_size * B, embed_dim]
    """
    if not is_dist_avail_and_initialized():
        return tensor
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ckpt(path, model, optimizer):
    ckpt = torch.load(path, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data