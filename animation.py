import torch
import torch.nn as nn
import numpy as np
from utils import get_training_imgs
import imageio


def animation(Gs, sigmas, animation_z, imgs, scale=4/3, device="cuda:0"):
    img = imgs[0]
    h, w = img.shape[0], img.shape[1]
    for i in range(len(Gs)):
        G = Gs[i]
        sigma = sigmas[i]
        if i < 1:
            z = animation_z
            prev = torch.zeros_like(animation_z).to(device)
        else:
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
    num_frame = 10
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imgs = get_training_imgs(img_path)
    checkpoint = torch.load(model_path)
    sigmas = checkpoint["sigmas"]
    Gs = checkpoint["Gs"]
    img = imgs[0]
    h, w = img.shape[0], img.shape[1]
    z0 = torch.randn(1, 3, h, w).to(device)
    z1 = torch.randn(1, 3, h, w).to(device)
    frames = []
    for k in range(num_frame):
        animation_z = z0 + (z1 - z0) * k / num_frame
        frame = animation(Gs, sigmas, animation_z, imgs, scale=4 / 3, device=device).cpu().detach().numpy()[0]
        gen = np.transpose(frame, axes=[1, 2, 0])
        frames.append(np.uint8((gen + 1) * 127.5))
    imageio.mimsave(img_path[:-4]+".gif", frames, 'GIF', duration=0.3)



