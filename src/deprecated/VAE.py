"""
This is a slight modification of Shubham Chandel's implementation of a
variational autoencoder in PyTorch.

Source: https://github.com/sksq96/pytorch-vae
"""
import cv2
import tqdm
import numpy as np
import torch

if __name__ == '__main__':
    # load img, data is in
    # https://drive.google.com/file/d/1RW3ewoS4FwXlCRVh4Dcb_n_xPniqHoeW/view?usp=sharing
    # with shape (10000, C=3, H=42, W=144), RGB format, 0~255
    imgs = np.load("./imgs.npy")
    # the original data is (384, 512, 3) RGB
    # first resize to (144, 144, 3)
    # then img = img[68:110, :, :]
    # finally transpose img to (N, C, H, W)
    # see vae.encode_raw for detail
    n = imgs.shape[0]
    indices = np.random.permutation(n)
    thres = int(n * 0.9)
    train_indices, test_indices = indices[:thres], indices[thres:]

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    bsz = 32
    lr = 1e-3
    vae = VAE().to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    num_epochs = 1000
    best_loss = 1e10
    for epoch in range(num_epochs):
        train_indices = np.random.permutation(train_indices)
        test_indices = np.random.permutation(test_indices)
        train_loss = []
        vae.train()
        for i in tqdm.trange(len(train_indices) // bsz, desc=f"Epoch #{epoch + 1} train"):
            index = train_indices[bsz * i: bsz * (i + 1)]
            img = torch.as_tensor(imgs[index] / 255., device=device, dtype=torch.float)
            loss = vae.loss(img, *vae(img))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        vae.eval()
        for i in tqdm.trange(len(test_indices) // bsz, desc=f"Epoch #{epoch + 1} test"):
            index = test_indices[bsz * i: bsz * (i + 1)]
            img = torch.as_tensor(imgs[index] / 255., device=device, dtype=torch.float)
            loss = vae.loss(img, *vae(img), kld_weight=0.)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        print(f'#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}')
        if test_loss < best_loss and epoch > num_epochs / 10:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(vae.state_dict(), 'vae.pth')
        # print imgs for visualization
        orig_img = torch.as_tensor(imgs[test_indices[0]] / 255., device=device, dtype=torch.float)
        vae_img = vae(orig_img[None])[0][0]
        # (C, H, W)/RGB -> (H, W, C)/BGR
        cv2.imwrite("orig.png", orig_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255)
        cv2.imwrite("vae.png", vae_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255)