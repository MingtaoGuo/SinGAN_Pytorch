from networks import Generator, Discriminator
from utils import get_training_imgs
from ops import train_single_scale
import torch


def train(path):
    imgs = get_training_imgs(path)
    nums = len(imgs)
    Gs = []
    Ds = []
    fixed_Zs = []
    sigmas = []
    ch = 16
    for i in range(nums):
        if i % 4 == 0:
            ch = ch * 2
        G = Generator(ch)
        D = Discriminator(ch)
        G.to("cuda:0")
        D.to("cuda:0")
        if i > 0:
            try:
                G.load_state_dict(G_.state_dict())
                D.load_state_dict(D_.state_dict())
                del G_, D_
            except:
                pass
        Gs.append(G)
        Ds.append(D)
        print(".............Total Scale: %d, current scale: %d............."%(nums, i+1))
        G_, D_ = train_single_scale(Gs, Ds, imgs[:i+1], sigmas, fixed_Zs)
    state_dict = {}
    state_dict["Gs"] = Gs
    state_dict["sigmas"] = sigmas
    state_dict["imgs"] = imgs
    torch.save(state_dict, path[:-3]+"pth")

if __name__ == "__main__":
    path = "./fire.jpg"
    train(path)