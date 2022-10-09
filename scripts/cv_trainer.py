import numpy as np
import tqdm
import torch
import cv2

from src.config.parser import read_config
from src.config.schema import cv_trainer_schema
import os
from src.config.yamlize import create_configurable, NameToSourcePath
import argparse

if __name__ == "__main__":
    # TODO: data augmentation
    parser = argparse.ArgumentParser(description='CV Trainer')
    parser.add_argument('git_repository', type=str,
                        help='repository for git')
    parser.add_argument('git_commit', type=str,
                        help='branch/commit for git')
    parser.add_argument('yaml_dir', type=str,
                        help='ex. ../config_files/train_vae')
    parser.add_argument('--wandb_key', type=str, dest='wandb_key',
                        help='api key for weights and biases')                        

    args = parser.parse_args()

    training_config = read_config(
        f"{args.yaml_dir}/training.yaml", cv_trainer_schema
    )

    with open(
        f"{training_config['model_save_path']}/git_config",
        "w+",
    ) as f:
        f.write(str(args))

    if not os.path.exists(f"{training_config['model_save_path']}"):
        os.umask(0)
        os.makedirs(training_config["model_save_path"], mode=0o777, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bsz = training_config["batch_size"]
    lr = training_config["lr"]
    encoder = create_configurable(
        f"{args.yaml_dir}/encoder.yaml", NameToSourcePath.encoder
    ).to(device)

    data_fetcher = create_configurable(
        f"{args.yaml_dir}/data_fetcher.yaml", NameToSourcePath.encoder_dataloader
    ).to(device)
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    num_epochs = training_config["num_epochs"]
    best_loss = 1e10

    train_ds, val_ds, train_dl, val_dl = data_fetcher.get_dataloaders(
        bsz,
        device,
    )

    # this is a stopgap - get_expert_demo_dataloaders has 1 value (autoencoding objective)
    # get_expert_demo_dataloaders has 1 value (autoencoding objective)
    num_inputs = len(val_ds[0])

    for epoch in range(num_epochs):
        train_loss = []
        encoder.train()
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch #{epoch + 1} train"):
            if num_inputs > 1:
                x = batch[:-1]
                y = batch[-1]
                loss = encoder.loss(y, encoder(x))
            else:
                loss = encoder.loss(batch, encoder(batch))
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        encoder.eval()
        for batch in tqdm.tqdm(val_dl, desc=f"Epoch #{epoch + 1} test"):
            # vae had a kld_weight=0.0 here... but yeah not sure how to parameterize that
            loss = encoder.loss(batch, encoder(batch)) 
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)
        print(f"#{epoch + 1} train_loss: {train_loss:.6f}, test_loss: {test_loss:.6f}")
        if test_loss < best_loss:
            best_loss = test_loss
            print(f"save model at epoch #{epoch + 1}")
            torch.save(
                encoder.state_dict(), f"{training_config['model_save_path']}/best.pth"
            )
