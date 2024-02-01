import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np
from typing import List


# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
cifar100_classes = cifar100.classes

def classify_with_clip(
        image_paths: List[str],
        classes: List[str],
) -> List[int]:

    if len(image_paths) == 0:
        return []    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(preprocess(image))

    image_input = torch.tensor(np.stack(images)).cuda()
    text_tokens = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    class_ids = np.argmax(similarity, axis=1).tolist()
    return class_ids


def demo():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    # Prepare the inputs
    image, class_id = cifar100[3637]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")



if __name__ == '__main__':
    class_ids = classify_with_clip(
        ['images/cup.png', 'images/horse.png'],
        ['horse', 'cup'],
    )

    print(class_ids)
