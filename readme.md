

# Latent Space Manipulation Project – StyleGAN2

This project was conducted by three students for a course at Télécom Paris. Feel free to contact any of use: [NathanRouille](https://github.com/NathanRouille) [Mamannne](https://github.com/Mamannne) [sachkho](https://github.com/sachkho/sachkho).

The main goal of this project was to give a semantic meaning to the latent space of a pretrained StyleGAN2 model. We implemented several methods from the literature, including InterfaceGAN, GANSpace, and Style Mixing. We also included a script to project real images into the latent space. The model we used was trained on the FFHQ dataset (faces) and was provided by NVIDIA: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl. Some of the scripts were adapted from the official StyleGAN2-ADA-PyTorch repository: https://github.com/NVlabs/stylegan2-ada-pytorch

The following of the README describes how to run each of the provided scripts, their goals, and the available options.

---

## Prerequisites 

- Python 3.7+
- PyTorch (CUDA if available)
- Python packages:
  ```bash
  pip install torch torchvision numpy scipy scikit-learn pillow click
  ```
* Clone or unzip the repository and move to the project root.

Note: Some scripts were not written by us. These are located in the `stylegan` folder and come from the official StyleGAN2-ADA-PyTorch repository.

---

## 1. `generate_and_classify.py`

**Purpose:**  
– Generates a large number of images by random sampling in the \$Z\$ space, maps them into \$W\$, synthesizes images, then computes and saves attribute scores using a pre-trained classifier.

**Usage:**

```bash
python generate_and_classify.py \
  --network https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
  --num-images 100 \
  --outdir data
```

**Options:**

* `--network`: path to the StyleGAN2 pickle (e.g., `ffhq.pkl`).
* `--num-images`: total number of \$z\$ vectors to generate.
* `--outdir`: output folder for JSON/NPZ score files.

---

## 2. `gram_schmidt.py`

**Purpose:**  
– Orthogonalizes a set of vectors (attribute directions) using the Gram–Schmidt process, to reduce semantic entanglement. It uses the file containing normal latent space directions obtained from SVMs (i.e., `boundary10000.json`) and saves the orthonormal directions in a JSON file (`basis10000.json`).

**Usage:**

```bash
python gram_schmidt.py 
```

---

## 3. `spacegan.py`

**Purpose:**  
– Implements the GANSpace method: fits a PCA on a set of \$W\$ codes, then generates images along the principal directions across selected layer ranges.

**Usage:**

```bash
python spacegan.py 
```

---

## 4. `stylegan/style_mixing.py`

**Purpose:**  
– Generates a style-mixing image grid, combining the coarse styles of a set of “rows” with the fine styles of a set of “cols”.

**Usage:**

```bash
python stylegan/style_mixing.py \
  --network https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
  --rows 1203,3405,34,902,910 \
  --cols 8023,122,67,7161 \
  --styles 0-6 \
  --trunc 0.7 \
  --noise-mode const \
  --outdir out/style_mixing
```

**Options:**

* `--network`: path to the StyleGAN2 pickle.
* `--rows`: list or range of seeds for rows (e.g., `85,100-105`).
* `--cols`: list or range of seeds for columns.
* `--styles`: range of layers to mix (`start-end`).
* `--trunc`: truncation value \$\psi\$.
* `--noise-mode`: `const` / `random` / `none`.
* `--outdir`: output folder for PNGs and the final grid.

---

## 5. `interfaceGAN.py`

**Purpose:**  
– Loads attribute directions (SVM + Gram–Schmidt) and/or centroids, then generates edited images along these directions (single or big plot), either from a synthetic latent code or via inversion of a real image.

**Usage:**

```bash
python interfaceGAN.py \
 --plot=single \
 --attribute=Eyeglasses \
 --filename=data/basis10000.json \
 --save=True \
 --proj=True \
 --img_proj=data/nathan.png \
 --auto_alpha=True
```

**Options:**

* `--attribute`: name of the attribute to edit (e.g., `Male`, `Smiling`, `Blond_Hair`, …).
* `--filename`: JSON file containing orthonormalized directions (boundaries).
* `--plot`: `single` (1 image) or `big` (2×4 grid of 7 alphas).
* `--alpha`: α step for `single` (linear displacement).
* `--auto_alpha`: `True` to compute α automatically using the centroid.
* `--seed`: seed of the latent code or synthetic reference image.
* `--start`, `--end`: α bounds for `big` (defines 7 linear values).
* `--noise_mode`: `const` / `random` / `none` for synthesis.
* `--save`: `True` to save the images & grid (`out/`).
* `--proj`: `True` to project a real image (`--img_proj`) before editing.
* `--img_proj`: path to the real image to project (e.g., `data/input.jpg`).

---

**SVM training option**  
If you want to (re)compute directions from existing latents and scores, set `train = True` at the top of the script, then run:

```bash
python interfaceGAN.py
```

This `train` mode will:

1. Load `data/ws10000.npy` and `data/metadata.json`
2. For each attribute listed in `ATTRIBUTES`, train an SVM on the top/bottom 1%
3. Save the boundaries (`data/boundary10000.json`) and centroids (`data/mu_pos10000.json`)


Here’s an extra README section you can add for `projector.py`, matching the style of the others:

---

## 6. `projector.py`

**Purpose:**
– Projects a given real image into the latent space $W$ of a pretrained StyleGAN2 network by optimizing $w$ so that the synthesized image matches the target in LPIPS (VGG16) feature space. Produces the final reconstructed image and the corresponding latent code.

**Usage:**

```bash
python projector.py \
  --network https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl \
  --target data/input.jpg \
  --outdir out/projection \
  --num-steps 1000 \
  --seed 303 \
  --save-video True
```

**Options:**

* `--network` (`network_pkl`) : path/URL to the pretrained StyleGAN2-ADA pickle (`G_ema` is used).
* `--target` (`target_fname`) : path to the input image to project (RGB). The script center-crops to a square and resizes to the generator’s resolution.
* `--outdir` : output directory for results.
* `--num-steps` : number of optimization steps (default: `1000`).
* `--seed` : random seed for initialization (default: `303`).
* `--save-video` : whether to save an MP4 showing optimization progress (default: `True`).
  **Note:** in the provided script, video export is currently disabled by an internal `if False:` guard—set this to `True` in the code if you want the video.

**Outputs (in `--outdir`):**

* `target.png` — the preprocessed (cropped & resized) target image used for projection.
* `proj.png` — the final synthesized image from the optimized latent.
* `projected_w.pkl` — the optimized latent $\mathbf{w}$ (shape `[num_ws, C]`) saved with `torch.save(...)`. You can reuse this for edits or further synthesis.

**Details & Notes:**

* Uses average $\mathbf{w}$ and its std computed from 10k random $\mathbf{z}$ samples to initialize and noise-schedule the optimization.
* LPIPS distance (via a VGG16 model downloaded at runtime) is minimized w\.r.t. $\mathbf{w}$ and per-layer noise buffers, with noise regularization.
* Expects CUDA: the script sets `device = torch.device('cuda')`. If you need CPU support, modify this line accordingly (it will be very slow).
* Input image size doesn’t need to match the generator; the script auto-crops to a centered square and resizes to `G.img_resolution`.
* To visualize the optimization trajectory, enable the video block in the code (replace `if False:` with `if save_video:`).


