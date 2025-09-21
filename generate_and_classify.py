########################################## Attention /!\ ##########################################
# Ce script ne peut se lancer qu'avec le folder "classifier" dans le même répertoire que ce script.
# Ce folder n'est pas dans le dépôt car trop lourd.
# Dans le cas où vous n'avez pas ce folder, vous allez générer des images mais sans classification.



import json
from tqdm import tqdm
from pathlib import Path
from sklearn import svm


import os
import re
from typing import List, Optional

import click

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import sys

sys.path.append(str(Path(__file__).parent.parent))
import stylegan.dnnlib as dnnlib
import stylegan.legacy as legacy

clf_flag = False

if (Path(__file__).parent / "classifier").is_dir():
    clf_flag = True
    from classifier.classification import get_classifier
    from classifier.constant import ATTRIBUTES


# Modifiez la fonction generate_images ou créez une nouvelle fonction
def generate_and_classify(
    network_pkl: str,
    outdir: str,
    num_images: int = 5000,
    truncation_psi: float = 1.0,
    noise_mode: str = 'const',
    batch_size: int = 8,
    save_img: bool = False,
    wd_file: str = "ws10000_test.npy"
):
    """Génère des images et les classe automatiquement"""
    # Charger le modèle de génération
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    ws = np.empty((5000,512), dtype=np.float32)
    
    # Charger le classifieur
    if clf_flag:
        print("Chargement du classifieur...")
        clf = get_classifier().to(device)
        clf.eval()
        print("Classifieur chargé.")
    else:
        print("\n\n\n /!\ Aucun classifieur trouvé. Le script va seulement générer des images sans classification. /!\ \n\n\n ")
        clf = None
    
    
    # Créer la structure de dossiers
    base_dir = Path(outdir)
    base_dir.mkdir(exist_ok=True)
    (base_dir / 'images').mkdir(exist_ok=True)
    
    metadata = []
    
    # Préparation des labels (si conditionnel mais ici non utilisé)
    label = torch.zeros([1, G.c_dim], device=device)
    
    # Génération par batch pour plus d'efficacité
    for i in tqdm(range(0, num_images, batch_size), desc="Generating images"):

        current_batch = min(batch_size, num_images - i)
        
        # Génération des images
        # Prepare seeds for this batch
        batch_seeds = range(i, i + current_batch)
        
        # Génération des vecteurs latents z
        z = torch.cat([
            torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            for seed in batch_seeds
        ])
        imgs = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # Calcul des vecteurs w
        w = G.mapping(z, None)
        
        ws[i:i + current_batch] = w[:, 0, :].cpu().numpy()
        
        # Classification
        with torch.no_grad():
            # Préparation de l'input pour le classifieur
            input_imgs = imgs.float() / 255.0  # Conversion [0,255] -> [0,1]
            input_imgs = input_imgs.permute(0, 3, 1, 2)  # Changement de format HWC -> CHW
            input_imgs = F.interpolate(input_imgs, size=224)  # Resize pour le classifieur
            
            # Prédiction
            if clf is not None:
                logits = clf(input_imgs)
                probs = torch.sigmoid(logits).cpu().numpy()

        # Sauvegarde des images et métadonnées
        for j in range(current_batch):
            seed = i + j
            img_path = base_dir / 'images' / f'seed{seed:04d}.png'
            
            # Sauvegarde de l'image
            if save_img:
                PIL.Image.fromarray(imgs[j].cpu().numpy(), 'RGB').save(img_path)
            
            # Création des métadonnées
            if clf is not None:
                class_probs = {ATTRIBUTES[k]: float(probs[j][k]) for k in range(len(ATTRIBUTES))}
                top_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)[:5]

                metadata.append({
                    'seed': seed,
                'image_path': str(img_path.relative_to(base_dir)),
                'class_probabilities': class_probs,
                'top_classes': dict(top_classes)
            })
    
    # Sauvegarde des métadonnées
    if clf is not None:
        with open(base_dir / 'metadata_test.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    np.save(base_dir / 'ws_test.npy', ws)
    
    print(f"Génération terminée. {num_images} images créées dans {outdir}")


@click.command()
@click.option('--network', 'network_pkl', default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', help='Modèle StyleGAN2')
@click.option('--outdir', required=True, help='Dossier de sortie')
@click.option('--num-images', type=int, default=1000, help='Nombre d\'images à générer')
def generate_and_classify_cli(network_pkl, outdir, num_images):
    generate_and_classify(network_pkl, outdir, num_images)

if __name__ == "__main__":
    generate_and_classify_cli()  