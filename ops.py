import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from PIL import Image

def gradient_penalty(x_real, x_fake, D, lambda_=10):
    eps = torch.rand(1, 1, 1, 1).to("cuda:0")
    x_hat = eps * x_real + (1. - eps) * x_fake
    x_hat = autograd.Variable(x_hat, requires_grad=True)
    outputs = D(x_hat)
    grads = autograd.grad(outputs, x_hat, torch.ones_like(outputs), retain_graph=True, create_graph=True)[0]
    penalty = lambda_ * ((torch.norm(grads, p=2, dim=1) - 1) ** 2).mean()
    return penalty

def concat_prevG(Gs, fixed_Zs, sigmas, imgs, is_rec=False):
    for i, G in enumerate(Gs):
        z_h, z_w = fixed_Zs[i].size()[2], fixed_Zs[i].size()[3]
        prev_h, prev_w = imgs[i+1].shape[0], imgs[i+1].shape[1]
        if i == 0:
            if is_rec:
                z = fixed_Zs[i]
            else:
                z = torch.randn(1, 3, z_h, z_w).to("cuda:0")
            prev = G(z, torch.zeros_like(z).to("cuda:0"))
            upsample = nn.Upsample((prev_h, prev_w))
            next_ = upsample(prev)
        else:
            if is_rec:
                z = torch.zeros(1, 3, z_h, z_w).to("cuda:0")#fixed_Zs[i]
            else:
                z = torch.randn(1, 3, z_h, z_w).to("cuda:0") * sigmas[i]
            prev = G(z, next_.detach())
            upsample = nn.Upsample((prev_h, prev_w))
            next_ = upsample(prev)

    return next_

def train_single_scale(Gs, Ds, imgs, sigmas, fixed_Zs):
    img = imgs[-1]
    img = np.transpose(img, axes=[2, 0, 1])[np.newaxis]
    real_img = torch.tensor(img/127.5-1.0, dtype=torch.float32).to("cuda:0")
    G = Gs[-1]
    D = Ds[-1]
    Opt_D = torch.optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.999))
    Opt_G = torch.optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=Opt_D, milestones=[1600], gamma=0.1)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=Opt_G, milestones=[1600], gamma=0.1)
    mse = nn.MSELoss()
    if len(Gs) == 1:
        for i in range(2000):
            sigma = 1.
            z = torch.randn(img.shape).to("cuda:0") * sigma
            z_rec = torch.randn(img.shape).to("cuda:0") * sigma
            for j in range(3):
                Opt_D.zero_grad()
                fake_img = G(z, torch.zeros_like(z).to("cuda:0"))
                fake_logits = D(fake_img.detach())
                real_logits = D(real_img)
                D_loss = fake_logits.mean() - real_logits.mean() + gradient_penalty(real_img, fake_img, D, lambda_=0.1)
                D_loss.backward(retain_graph=True)
                Opt_D.step()
            for j in range(3):
                Opt_G.zero_grad()
                fake_img = G(z, torch.zeros_like(z).to("cuda:0"))
                fake_logits = D(fake_img)
                rec = mse(G(z_rec, torch.zeros_like(z_rec)), real_img)
                G_loss = -fake_logits.mean() + rec * 10
                G_loss.backward(retain_graph=True)
                Opt_G.step()
            if i % 100 == 0:
                fake_img = fake_img.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
                Image.fromarray(np.uint8((fake_img + 1) * 127.5)).save("./results/" + str(fake_img.shape[0]) + "_" + str(i) + ".jpg")
                print("Iteration: %d, D_loss: %f, G_loss: %f" % (i, D_loss, G_loss))
            schedulerD.step()
            schedulerG.step()
        fixed_Zs.append(z_rec)

    else:
        for i in range(2000):
            init_sigma = 0.1
            z_ = torch.randn(1, 3, img.shape[2], img.shape[3]).to("cuda:0")
            for j in range(3):
                Opt_D.zero_grad()
                prev_x = concat_prevG(Gs[:-1], fixed_Zs, sigmas, imgs, is_rec=False)
                prev_z = concat_prevG(Gs[:-1],  fixed_Zs, sigmas, imgs, is_rec=True)
                sigma = torch.sqrt(mse(prev_z, real_img)) * init_sigma
                z = sigma * z_
                fake_img = G(z, prev_x)
                fake_logits = D(fake_img.detach())
                real_logits = D(real_img)
                D_loss = fake_logits.mean() - real_logits.mean() + gradient_penalty(real_img, fake_img, D, lambda_=0.1)
                D_loss.backward(retain_graph=True)
                Opt_D.step()
            for j in range(3):
                Opt_G.zero_grad()
                fake_logits = D(fake_img)
                rec = mse(G(torch.zeros_like(prev_z).to("cuda:0"), prev_z), real_img)
                G_loss = -fake_logits.mean() + rec * 10
                G_loss.backward(retain_graph=True)
                Opt_G.step()
            if i % 100 == 0:
                fake_img = fake_img.permute(0, 2, 3, 1).cpu().detach().numpy()[0]
                Image.fromarray(np.uint8((fake_img + 1) * 127.5)).save("./results/" + str(fake_img.shape[0]) + "_" + str(i) + ".jpg")
                print("Iteration: %d, D_loss: %f, G_loss: %f" % (i, D_loss, G_loss))
            schedulerD.step()
            schedulerG.step()
        fixed_Zs.append(torch.zeros_like(z))

    sigmas.append(sigma)
    return G, D


    pass