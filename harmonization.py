import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from utils import get_training_imgs


def harmonization(Gs, sigmas, imgs, scale=4/3, n=1, device="cuda:0"):
    img = imgs[n]
    img = np.transpose(img, axes=[2, 0, 1])[np.newaxis] / 127.5 - 1.0
    prev = torch.tensor(img, dtype=torch.float32).to(device)
    h, w = img.shape[2], img.shape[3]
    for i in range(n, len(Gs)):
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
    img_path = "./star_cat.jpg"
    model_path = "./star.pth"
    n = 5 # different scale number has different harmonization resutls
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imgs = get_training_imgs(img_path)
    checkpoint = torch.load(model_path)
    sigmas = checkpoint["sigmas"]
    Gs = checkpoint["Gs"]
    gen = harmonization(Gs, sigmas, imgs, n=n, device=device).cpu().detach().numpy()[0]
    gen = np.transpose(gen, axes=[1, 2, 0])
    Image.fromarray(np.uint8((gen+1)*127.5)).show()

