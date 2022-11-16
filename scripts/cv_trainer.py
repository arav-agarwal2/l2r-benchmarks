import numpy as np
import tqdm
import torch
import cv2

from src.config.parser import read_config
from src.config.schema import cv_trainer_schema
import os
from src.config.yamlize import create_configurable, NameToSourcePath
import argparse
from src.constants import DEVICE
import logging

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
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p', 
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(f"{training_config['model_save_path']}", 'debug.log')),
            logging.StreamHandler()
        ],
        force=True
    )
    if not torch.cuda.is_available():
        logging.info("cuda not available, exiting")
        sys.exit(-1)

    if not os.path.exists(training_config['model_save_path']):
        os.mkdir(training_config['model_save_path'])
    
    with open(
        f"{training_config['model_save_path']}/git_config",
        "w+",
    ) as f:
        f.write(str(args))

    if not os.path.exists(f"{training_config['model_save_path']}"):
        os.umask(0)
        os.makedirs(training_config["model_save_path"], mode=0o777, exist_ok=True)

    bsz = training_config["batch_size"]
    lr = training_config["lr"]
    encoder = create_configurable(
        f"{args.yaml_dir}/encoder.yaml", NameToSourcePath.encoder
    ).to(DEVICE)

    data_fetcher = create_configurable(
        f"{args.yaml_dir}/data_fetcher.yaml", NameToSourcePath.encoder_dataloader
    )
    optim = torch.optim.Adam(encoder.parameters(), lr=lr)
    num_epochs = training_config["num_epochs"]
    best_loss = 1e10

    train_ds, val_ds, train_dl, val_dl = data_fetcher.get_dataloaders(
        bsz,
        DEVICE,
    )

    # this is a stopgap - get_expert_demo_dataloaders has 1 value (autoencoding objective)
    # get_expert_demo_dataloaders has 1 value (autoencoding objective)
    multiple_inputs = type(val_ds[0]) == tuple

    for epoch in range(num_epochs):
        train_loss = []
        encoder.train()
        for batch in tqdm.tqdm(train_dl, desc=f"Epoch #{epoch + 1} train"):
            optim.zero_grad()
            if multiple_inputs:
                # todo expand to more than 1, or 2 things passed by dataloader?
                x = batch[0]
                y = batch[-1]
                cv2.imwrite(f"{training_config['model_save_path']}/debug_input{epoch+1}.png", x[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
                pred = encoder(x)
                out_mask = torch.argmax(pred, dim=1)[0]
                cv2.imwrite(f"{training_config['model_save_path']}/debug_output{epoch+1}.png", out_mask.detach().cpu().numpy() * 255)
                loss = encoder.loss(y, pred)
            else:
                loss = encoder.loss(batch, encoder(batch))
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        test_loss = []
        encoder.eval()
        for batch in tqdm.tqdm(val_dl, desc=f"Epoch #{epoch + 1} test"):
            # vae had a kld_weight=0.0 here... but yeah not sure how to parameterize that
            if multiple_inputs:
                # todo expand to more than 1, or 2 things passed by dataloader?
                x = batch[0]
                y = batch[-1]
                loss = encoder.loss(y, encoder(x))
            else:
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
        torch.save(
            encoder.state_dict(), f"{training_config['model_save_path']}/epoch{epoch + 1}.pth"
        )
