# l2r-benchmarks

To run this branch - simply do the following:

Find safepo.yaml on my (Arav's) hot storage. Copy it to your hot storage, along with the Arrival simulator.

```
kubectl create -f safepo.yaml
kubectl exec --it safepo -- /bin/bash
```

Then run the following to setup the container:

```
cd /mnt
git clone this repo
git checkout this branch
cd l2r-benchmarks
chmod +x runme.sh
./runme.sh
python3.8 - scripts.main-aicrowd
