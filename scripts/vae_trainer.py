import numpy as np
import tqdm
import torch
import cv2

from src.encoders.vae import VAE
from src.encoders.dataloaders.expert_demo_dataloader import get_expert_demo_dataloaders
from src.config.parser import read_config
from src.config.schema import cv_trainer_schema
import os


if __name__ == '__main__':
    # TODO: data augmentation

    training_config = read_config(
        "src/config_files/train_vae/training.yaml", cv_trainer_schema
    )

    if not os.path.exists(f"{training_config['model_save_path']}"):
            os.umask(0)
            os.makedirs(training_config['model_save_path'], mode=0o777, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    bsz = training_config["batch_size"]
    lr = training_config["lr"]
    vae = VAE().to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    num_epochs = 3
    best_loss = 1e10

    train_ds, val_ds, train_dl, val_dl = get_expert_demo_dataloaders("train/", "val/", device)

    for epoch in range(num_epochs):
        train_loss = []
        vae.train()
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch #{epoch + 1} train"):
            loss = vae.loss(batch, *vae(batch))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        vae.eval()
        for batch in tqdm.tqdm(val_dl, desc=f"Epoch #{epoch + 1} test"):
            loss = vae.loss(batch, *vae(batch), kld_weight=0.)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        print(f'#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}')
        if test_loss < best_loss and epoch > num_epochs / 10:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(vae.state_dict(), f"{training_config['model_save_path']}/vae_{epoch + 1}.pth")

        # print imgs for visualization
        # orig_img = torch.as_tensor(val_ds[0], device=device, dtype=torch.float)
        # vae_img = vae(orig_img[None])[0][0]
        # (C, H, W)/RGB -> (H, W, C)/BGR
        # cv2.imwrite("orig.png", orig_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255) 
        # cv2.imwrite("vae.png", vae_img.detach().cpu().numpy()[::-1].transpose(1, 2, 0) * 255)