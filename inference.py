from model.clip import CLIP
from utils import load_yaml, load_json
from tokenizer import load_vocab, build_tokenizer

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def encode_all_images(model, dataset_json, image_root, transform):
    """
    return:
        image_feats shape: [N, embed_dim], N represents the number of images in the dataset
        image_paths: list of image paths
    """
    image_feats = []
    image_paths = []

    data = load_json(dataset_json)

    with torch.no_grad():
        for item in data:
            img_path = f"{image_root}/{item['image']}"
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).cuda()

            feat = model.image_encoder(image)  # [1, D]
            feat = F.normalize(feat, dim=-1)

            image_feats.append(feat)
            image_paths.append(img_path)

    image_feats = torch.cat(image_feats, dim=0)
    return image_feats, image_paths


def retrieve_image(model, text, tokenizer, image_feats, image_paths, topk=1):
    """
    image_feats shape: [N, embed_dim]
    image_paths: list of image paths
    return: the path of the retrieved result image
    """
    text_ids = torch.tensor(tokenizer(text), dtype=torch.long).unsqueeze(0).cuda()

    with torch.no_grad():
        text_feat = model.text_encoder(text_ids)    # [1, embed_dim]
        text_feat = F.normalize(text_feat, dim=-1)

        sim = text_feat @ image_feats.T             # [1, N]
        values, indices = sim.topk(topk, dim=-1)

    return image_paths[indices[0]]


def show_images(image_path):
    """
    image_paths: the path of the retrieved result image
    """
    img = Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


config = load_yaml("configs/inference_config.yaml")
model = CLIP(**config["model"]).cuda()
pretrained = torch.load(config["pretrained"], map_location="cuda")
print(f"The number of epochs that have been trained: {pretrained['epoch']}")
model.load_state_dict(pretrained["model"])
model.eval()
tokenizer = build_tokenizer(load_vocab(config["vocab_path"]))

transform = transforms.Compose([transforms.Resize(config["transforms"]["Resize"]),
                                transforms.CenterCrop(config["transforms"]["CenterCrop"]),
                                transforms.ToTensor()])
image_feats, image_paths = encode_all_images(model, dataset_json=config["json_path"], image_root=config["image_root"],
                                             transform=transform)
query = "A bicycle replica with a clock as the front wheel."
results = retrieve_image(model, query, tokenizer, image_feats, image_paths)
show_images(results)
