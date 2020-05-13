import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from utils import get_training_imgs


def random_sample_from_single(Gs, sigmas, h, w, scale=4/3, device="cuda:0"):
    prev = torch.zeros(1, 3, h, w).to(device)
    for i in range(len(Gs)):
        G = Gs[i]
        sigma = sigmas[i]
        z = torch.randn(1, 3, h, w).to(device) * sigma
        prev = G(z, prev)
        if i == len(Gs) - 1:
            break
        h, w = int(h * scale), int(w * scale)
        upsample = nn.Upsample((h, w))
        prev = upsample(prev)

    return prev

if __name__ == "__main__":
    img_path = "./star.jpg"
    model_path = "./star.pth"
    h, w = 50, 25
    imgs = get_training_imgs(img_path)
    checkpoint = torch.load(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sigmas = checkpoint["sigmas"]
    Gs = checkpoint["Gs"]
    gen = random_sample_from_single(Gs, sigmas, h, w, device=device).cpu().detach().numpy()[0]
    gen = np.transpose(gen, axes=[1, 2, 0])
    Image.fromarray(np.uint8((gen+1)*127.5)).show()

