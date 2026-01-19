from tokenizer import load_vocab, build_tokenizer

import json
import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class CLIPDataset(Dataset):
    def __init__(self, image_root, vocab_path, json_path, image_transform=None):
        self.image_root = image_root
        self.tokenizer = build_tokenizer(load_vocab(vocab_path))
        self.image_transform = image_transform
        with open(json_path, "r", encoding="utf-8") as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        item = self.datas[idx]

        image_path = os.path.join(self.image_root, item["image"])
        images = Image.open(image_path).convert("RGB")
        if self.image_transform is not None:
            images = self.image_transform(images)

        caption = item["caption"]
        token_ids = self.tokenizer(caption)

        return {"images": images, "text_ids": torch.tensor(token_ids, dtype=torch.long)}


def collate_fn(batch, pad_token_id=0):
    """
    batch: List[{"image": Tensor, "text_ids": LongTensor}]
    return:
        images:        [B, C, H, W]
        text_ids:      [B, L_max]
        attention_mask:[B, L_max]
    """
    images = torch.stack([item["images"] for item in batch], dim=0)

    text_ids_list = [item["text_ids"] for item in batch]
    lengths = [len(t) for t in text_ids_list]
    max_len = max(lengths)

    B = len(batch)
    text_ids = torch.full((B, max_len), fill_value=pad_token_id, dtype=torch.long)

    for i, ids in enumerate(text_ids_list):
        l = ids.size(0)
        text_ids[i, :l] = ids

    return {"images": images, "text_ids": text_ids}


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    image_root = "data/train_image"
    vocab_path = "data/vocab.txt"
    json_path = "data/train_image_caption.json"
    transform = transforms.Compose([transforms.Resize(480), transforms.CenterCrop(480), transforms.ToTensor()])
    dataset = CLIPDataset(image_root, vocab_path, json_path, transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    num = 0
    for batch in dataloader:
        print(f"image shape: {batch['images'].shape}")
        print(f"caption shape: {batch['text_ids'].shape}")
        num += 1
        if num == 5: break