# l2r-benchmarks

To run this branch - simply do the following:

Find safepo.yaml on my (Arav's) hot storage. Copy it to your hot storage, along with the Arrival simulator.

```
kubectl create -f safepo.yaml
kubectl exec -it safepo -- /bin/bash
```

Then run the following to setup the container:

```
cd /mnt
sudo apt-get update
sudo apt-get install git
git clone this repo
cd l2r-benchmarks
git checkout this branch
chmod +x runme.sh
./runme.sh
python3.8 -m scripts.main-aicrowd
