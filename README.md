# Crypson

Modern cryptography algorithms using deep learning


```bash
conda activate crypson
```

```bash
cd extern; pip install -e vector-quantize-pytorch
```


```bash
python train.py --batch-size 128 --num-epochs 200 --es-patience 1000 --val-split 0.05 --z-dim 64 --resolution 32 --gpus 0 --dataset '../../data'

python test.py --model-dir './train/DCGan' --prompt-path '../../demo/gan.txt' --classes-path '../../data/idx_to_class.json' --output-dir '../../demo/results' --latent-dim 64

python train.py --batch-size 128 --num-workers 8 --generator '../../checkpoints/gan/epoch_00199-loss_0.63360.ckpt' --train-size 47000 --test-size 3000 --latent-dim 8 --kl-w 0.7 --num-epochs 100 --hidden-channels 32 64 128 256



python train.py --batch-size 128 --num-workers 8 --generator '../../checkpoints/gan/epoch_00199-loss_0.63360.ckpt' --autoencoder '../../checkpoints/vae/epoch_00098-loss_7669.00684.ckpt' --train-size 94000 --test-size 6000 --latent-dim 8 --num-epochs 100 --hidden-channels 32 64 128 256

python test.py --batch-size 128 --num-workers 8 --generator '../../checkpoints/gan/epoch_00199-loss_0.63360.ckpt' --autoencoder '../../checkpoints/vae/epoch_00098-loss_7669.00684.ckpt' --classifier '../../checkpoints/classifier/epoch_00099-loss_0.90071.ckpt' --train-size 470 --test-size 6000 --latent-dim 8 --hidden-channels 32 64 128 256 --debug
```


### TODO

- [ ] Upload trained checkpoint from server to GitHub
- [x] Verify the test utility for cGAN
- [ ] Implement a quantization mechanism for the cGAN
- [ ] Implement a VQ-VAE
- [ ] Use tensorrt and deploy the two models on a Jetson Nano
- [ ] Create a server-client architecture to encrypt and decrypt files
- [ ] Create Dockerfile
- [ ] Create conda env files
- [ ] Complete `README.md` with instructions and experimental results
- [ ] Post my project report and demo video under `docs` directory
- [ ] Create a GAN `README.md`
- [ ] Create a VQ-VAE `README.md`
