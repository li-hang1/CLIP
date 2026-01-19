from model.clip import CLIP
from dataset import CLIPDataset, collate_fn
from loss import clip_loss
from utils import load_yaml, load_ckpt

import torch
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def main(config):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    is_main = (rank == 0)

    transform = transforms.Compose([transforms.Resize(config["transforms"]["Resize"]),
                                    transforms.CenterCrop(config["transforms"]["CenterCrop"]),
                                    transforms.ToTensor()])
    train_dataset = CLIPDataset(config["data"]["image_root"], config["data"]["vocab_path"], config["data"]["json_path"],
                                image_transform = transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=20,
                                  pin_memory=True, collate_fn=collate_fn, drop_last=True)
    model_config = config["model"]
    model = CLIP(**model_config).cuda()
    model = DDP(model, device_ids=[local_rank])
    optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9,0.999))
    start_epoch = 0

    ckpt_path = os.path.join(config["pretrained"], "best.pth")
    if config["resume"] and os.path.exists(ckpt_path):
        start_epoch = load_ckpt(ckpt_path, model, optimizer)
        if is_main:
            print(f"Resumed from {ckpt_path}, epoch: {start_epoch}")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        train_sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(train_dataloader):
            text_ids, images = batch["text_ids"].cuda(), batch["images"].cuda()
            text_features, image_features, logit_scale = model(text_ids, images)
            loss = clip_loss(image_features, text_features, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if is_main and batch_idx % 100 == 0:
                print(f"Epoch: {epoch} | batch_idx: {batch_idx} | Train loss: {loss:.4f}")
        torch.save({"model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                   os.path.join(config["pretrained"], "best.pth"))

    dist.destroy_process_group()


if __name__ == "__main__":
    config = load_yaml("configs/train_config.yaml")
    main(config)
