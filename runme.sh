sudo apt-get update
sudo apt-get install python3.8

python3.8 -m pip install --upgrade pip setuptools wheel
python3.8 -m pip install wandb strictyaml pyyaml jsonpickle
python3.8 -m pip install -r ../l2r-starter-kit/requirements.txt
