import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import stylegan.dnnlib as dnnlib
import stylegan.legacy as legacy


N = 10000

#on importe le modele ffhq

with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to('cuda')


#on définit l'opération E qui prends 
#les directions de la PCA qu'on veut imposer au vecteur de style,
#et les couches affectées par cette modification.

class E:
    def __init__(self,directions, num_vec,i=0,j=18):
        self.direction = num_vec
        self.v = directions[num_vec]
        self.i = i
        self.j = j


E.__repr__ = lambda self: f'E(v{self.direction},{self.i}-{self.j})'

def get_layerwise_directions(pca):
    """
    Renvoi une liste de n_comp directions, toutes sous forme de vecteur de taille [num_layers, w_dim].
    """
    dirs = []
    for v in pca.components_:
        # v de taille (L*D), reshape en (L, D)
        dirs.append(np.repeat(v[np.newaxis, :], 18, axis=0))
    return dirs

# On récupere les directions "layer-wise"
num_layers = G.mapping.num_ws 
w_dim      = G.z_dim          


def get_latent_codes(ws,seed,sigma,e):
    w = ws[seed].copy()  # On récupere le code latent de la seed
    w = np.repeat(w[np.newaxis, :], num_layers, axis=0)
    latent_codes = []
    for step in np.linspace(-8*sigma,8*sigma, num=7):
        w_edit = w.copy()
        w_edit[e.i:e.j] += e.v[e.i:e.j] * step
        latent_codes.append(w_edit)
    return np.array(latent_codes)

def plot_latent_codes(latent_codes, seed, sigma, e):
    fig, axes = plt.subplots(1, 7, figsize=(25, 3))
    fig.suptitle(f'Seed {seed} - {e}', fontsize=16)
    steps = np.linspace(-8*sigma, 8*sigma, num=7)
    for i, w in enumerate(latent_codes):
        img = G.synthesis(torch.from_numpy(w).unsqueeze(0).to('cuda'), noise_mode='const')
        img = (img + 1) / 2  # On normalise
        img = img.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)[0]
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Step {steps[i]:.2f} sigma', fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print('Loaded generator')

    ws = np.load('data/ws10000.npy', allow_pickle=True)

    print('Generated latent codes of size', len(ws) , 'with shape', ws[0].shape)

    try:
        pca = joblib.load('data/pca_ws10000.pkl')
        print('Loaded existing PCA model with', pca.n_components_, 'components')
    except FileNotFoundError:
        print('Fitting new PCA model...')

        pca = PCA(n_components=100)   #On choisit 100 composantes de la PCA comme indiqué dans l'article GANSpace
        pca.fit(ws)             

        joblib.dump(pca, 'data/pca_ws10000.pkl')

        print('Fitted PCA with', pca.n_components_, 'components')

    directions = get_layerwise_directions(pca)


    sigma = np.mean(np.std(ws, axis=0))  # On prends l'ecart type pour le déplacement
    import os

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs("out_spacegan", exist_ok=True)

    # Seeds et directions à explorer
    seeds = [24]
    direction_ids = range(5)
    layer_ranges = [(0, 18), (0, 8), (8, 18)]

    # Boucle principale
    for seed in seeds:
        for d_id in direction_ids:
            for (i, j) in layer_ranges:
                e = E(directions, d_id, i=i, j=j)
                latent_codes = get_latent_codes(ws, seed=seed, sigma=sigma, e=e)
                # Génère la figure
                fig, axes = plt.subplots(1, 7, figsize=(35, 5))
                fig.suptitle(f'Seed {seed} - {e}', fontsize=16)
                steps = np.linspace(-8 * sigma, 8 * sigma, num=7)
                for idx, w in enumerate(latent_codes):
                    img = G.synthesis(torch.from_numpy(w).unsqueeze(0).to('cuda'), noise_mode='const')
                    img = (img + 1) / 2
                    img = img.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)[0]
                    axes[idx].imshow(img)
                    axes[idx].axis('off')
                    axes[idx].set_title(f'Step {steps[idx]:.2f}σ', fontsize=12)

                plt.tight_layout(rect=[0, 0, 1, 0.95])

                # Enregistrement
                out_path = f"out_spacegan/edit_seed{seed}_v{d_id}_layers{i}-{j}.png"
                plt.savefig(out_path)
                print(f"✅ Saved: {out_path}")
                plt.close()
