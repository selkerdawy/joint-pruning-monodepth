# joint-pruning-monodepth
Lightweight Monocular Depth Estimation Model by Joint End-to-End Filter pruning.

# **Demo**
[Sample video](https://www.youtube.com/watch?v=gSkuJB7Or5w) showing the pruned vgg model and the baseline [monodepth vgg](https://github.com/mrharicot/monodepth "monodepth vgg") running on GTX 1080 Ti with 60 and 33 frame per second respectively; The demo is slowed down for demonstration only. The demo shows that even with more than 80% compression rate, the pruned network shows both qualitatively and quantitatively small drop in accuracy compared to the baseline network.

# Inference
Sample code for inference using the pruned vgg model trained on eigen split is provided, usage:
```bash
python sample_code.py --dir PATH/TO/KITTI/2011_09_26/2011_09_26_drive_0064_sync/image_02/data/ --checkpoint_path model/model-0.data-00000-of-00001
```

**Environment**
virtualenv is recommended and install requirements from req.txt:
```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r req.txt
```

Training code will be added soon.

# Supplementary materials
Full depth metrics disrcarded from original paper due to space, details on the number of filters per layer in the pruned network, and comparison between weights sparsity vs masks sparsity.
