import numpy as np
import torch
from sklearn import svm
from typing import Optional, Union
import json
import matplotlib.pyplot as plt
import click
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import stylegan.dnnlib as dnnlib
import stylegan.legacy as legacy
from projector import run_projection
from click.testing import CliRunner



def train_boundary(
    latent_codes: Union[np.ndarray, torch.Tensor],
    scores: Union[np.ndarray, torch.Tensor],
    chosen_num_or_ratio: float = 0.01,
    split_ratio: float = 0.7,
    invalid_value: Optional[float] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Entrainement d'un classifieur SVM pour trouver la frontière entre les échantillons positifs et négatifs pour un attribut donné.
    """
    # Convertir les tenseurs en array numpy si nécessaire
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    # Flatten les codes latent si nécessaire
    if latent_codes.ndim == 3:
        N, L, D = latent_codes.shape
        latent_codes = latent_codes.reshape(N, L * D)
    elif latent_codes.ndim == 2:
        N, D = latent_codes.shape
    else:
        raise ValueError("latent_codes must be a 2D or 3D array/tensor")

    # s'assurer de la dimension des scores
    scores = scores.reshape(-1)
    if scores.shape[0] != N:
        raise ValueError("scores must have the same number of samples as latent_codes")

    # filtrer les valeurs invalides
    if invalid_value is not None:
        mask = scores != invalid_value
        latent_codes = latent_codes[mask]
        scores = scores[mask]
        N = scores.shape[0]

    # trier les scores par ordre décroissant
    idx = np.argsort(scores)[::-1]
    latent_codes = latent_codes[idx]
    scores = scores[idx]

    # déterminer le nombre d'échantillons à choisir (positifs/négatifs)
    if 0 < chosen_num_or_ratio <= 1:
        chosen_num = int(N * chosen_num_or_ratio)
    else:
        chosen_num = int(chosen_num_or_ratio)
    chosen_num = min(chosen_num, N // 2)
    if chosen_num < 1:
        raise ValueError("chosen_num_or_ratio too small, no samples selected")
    
    # centroide des échantillons positifs (nécessaires pour la méthode de pas automatique)
    pos_latents = latent_codes[:chosen_num]
    mu_pos = pos_latents.mean(axis=0)

    # séparer les échantillons en positifs et négatifs en données d'entraînement et de validation
    np.random.seed(seed)
    # échantillons positifs: les chosen_num premiers
    pos_indices = np.arange(chosen_num)
    np.random.shuffle(pos_indices)
    train_pos = latent_codes[pos_indices[:int(chosen_num * split_ratio)]]
    val_pos   = latent_codes[pos_indices[int(chosen_num * split_ratio):chosen_num]]

    # Échantillons négatifs: les derniers chosen_num
    neg_indices = np.arange(N - chosen_num, N) - (N - chosen_num)
    np.random.shuffle(neg_indices)
    train_neg = latent_codes[N - chosen_num + neg_indices[:int(chosen_num * split_ratio)]]
    val_neg   = latent_codes[N - chosen_num + neg_indices[int(chosen_num * split_ratio):]]

    # Préparer les données d'entraînement et les étiquettes
    train_data = np.concatenate([train_pos, train_neg], axis=0)
    train_labels = np.concatenate([
        np.ones(len(train_pos), dtype=int),
        np.zeros(len(train_neg), dtype=int)
    ], axis=0)

    # Entraîner le classifieur SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_labels)

    # Valider le classifieur
    val_data = np.concatenate([val_pos, val_neg], axis=0)
    val_labels = np.concatenate([
        np.ones(len(val_pos), dtype=int),
        np.zeros(len(val_neg), dtype=int)
    ], axis=0)
    val_accuracy = clf.score(val_data, val_labels)
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Extraire et normaliser la frontière
    coef = clf.coef_.reshape(-1)
    boundary = coef / np.linalg.norm(coef)
    return torch.from_numpy(boundary).float(), mu_pos


def save_boundary(boundary: torch.Tensor, attribute: str,filepath: str = 'data/boundary.json') -> None:
    """
    Sauvegarde de la frontière (tenseur) dans un fichier JSON.
    """
    try :
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {} 

    data[attribute] = boundary.cpu().numpy().tolist()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def save_mu(mu_pos, attribute: str,filepath: str = 'data/mu_pos.json') -> None:
    """
    Sauvegarde de la position du centroïde (mu_pos) dans un fichier JSON.
    """

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {} 

    data[attribute] = mu_pos.tolist()

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_boundary(attribute: str,filename: str = 'data/boundary.json') -> np.ndarray:
    """
    Charge la frontière pour un attribut donné à partir d'un fichier JSON.
    """
    with open(filename, 'r') as f:
        boundaries = json.load(f)
    
    if attribute not in boundaries:
        raise ValueError(f"Boundary for attribute '{attribute}' not found.")
    
    boundary = np.array(boundaries[attribute])
    return boundary

def load_mu(attribute, filepath='data/mu_pos.json'):
    """Charge la position du centroïde (mu_pos) pour un attribut donné à partir d'un fichier JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    if attribute not in data:
        raise ValueError(f"mu_pos for '{attribute}' not in {filepath}")
    return np.array(data[attribute])

def compute_alpha(w_single, boundary, mu_pos, clip=5.0):
    """
    Calcule automatiquement le pas alpha
    """
    n = boundary.reshape(-1).astype(np.float64)
    n /= np.linalg.norm(n) # normalisation
    # projection du vecteur entre le point et le centroïde sur la direction normale à la frontière
    alpha = float(np.dot(n, mu_pos.reshape(-1) - w_single.reshape(-1)))
    return alpha


def big_plot(ws, seed, w_proj, boundary, ref, alpha_list=np.linspace(-5, 5, 7), noise_mode='const', save=False):
    """
    Plot une grille d'images éditées pour différentes valeurs d'alpha.
    """
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))

    if len(alpha_list) != 7:
        raise ValueError("alpha_list must contain exactly 7 values.")

    # Place l'image de référence dans la première case
    ax[0, 0].imshow(ref)
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Reference')

    images = []
    for i, alpha in enumerate(alpha_list):
        if w_proj is not None:
            # si on projette une image réelle, on utilise le code latent projeté
            w_edit = w_proj.copy()
        else:
            # sinon on utilise le code latent de la seed donnée
            w_edit = ws[seed].copy() 
        w_edit += alpha * boundary # déplacement dans la direction normale à la frontière du pas alpha

        w_edit = torch.from_numpy(w_edit).to('cuda')

        # Génération de l'image avec le code latent modifié
        with torch.no_grad():
            img = G.synthesis(w_edit.unsqueeze(0), noise_mode=noise_mode)  

        img_np = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        images.append(img_np)

        # Affichage de l'image dans la grille
        row = (i + 1) // 4
        col = (i + 1) % 4
        ax[row, col].imshow(img_np)
        ax[row, col].axis('off')
        ax[row, col].set_title(f'alpha={alpha:.2f}')

        if save:
            img_clipped = np.clip(img_np, 0, 255)
            plt.imsave(f'out/after_{i}.png', img_clipped.astype(np.uint8))

    plt.tight_layout()
    if save:
        # sauvegarde de la figure
        fig.savefig(f'out/grid.png')
    plt.show()


def single_plot(ws, seed, w_proj, boundary, ref,alpha,noise_mode='const',save=False):
    """
    Plot une seule image éditée avec le code latent modifié par un pas alpha.
    """
    if w_proj is not None:
        # si on projette une image réelle, on utilise le code latent projeté
        w_edit = w_proj.copy()
    else:
        # sinon on utilise le code latent de la seed donnée
        w_edit = ws[seed].copy() 
    w_edit += alpha * boundary  # déplacement dans la direction normale à la frontière du pas alpha

    w_edit = torch.from_numpy(w_edit).to('cuda')

    print('Generating image with edited latent code...')

    # Génération de l'image avec le code latent modifié
    with torch.no_grad():
        img = G.synthesis(w_edit.unsqueeze(0), noise_mode=noise_mode)


    img_np = (img.permute(0,2,3,1) * 127.5 + 128).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
    if save:
        img_clipped = np.clip(img_np, 0, 255)
        plt.imsave(f'out/edited_alpha{alpha:.2f}.png', img_clipped)

    # Affichage de l'image éditée avec la référence
    fig, ax = plt.subplots(1,2,figsize=(12, 6))
    ax[0].imshow(ref)
    ax[0].axis('off')
    ax[0].set_title('Reference Image')
    ax[1].imshow(img_np)
    ax[1].axis('off')
    ax[1].set_title(f'Edited Image (alpha={alpha})')
    plt.tight_layout()
    plt.show()


# attributs souhaités pour l'entraînement du classifieur SVM
ATTRIBUTES = ['Male', 'Smiling','Young','Eyeglasses','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Bald']

train = False  # True pour entraîner le classifieur SVM, False pour visualiser les résultats

if __name__ == "__main__":

    ws = np.load('data/ws10000.npy')  # Chargement des codes latents pré-calculés

    if train:

        ws = ws

        # charger les métadonnées pour obtenir les scores par attribut
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)


        print('Latent codes loaded')

        for ATTRIBUTE in ATTRIBUTES:

            # extraire les scores pour l'attribut donné
            scores = np.array([item['class_probabilities'][ATTRIBUTE] for item in metadata])


            latent_codes = np.stack(ws)

            # entrainement de la frontière du SVM et du centroïde pour l'attribut
            boundary, mu_pos = train_boundary(latent_codes, scores)

            # sauvegarde de la frontière et du centroïde
            save_boundary(boundary, ATTRIBUTE,'data/boundary10000.json')
            save_mu(mu_pos, ATTRIBUTE,'data/mu_pos10000.json')

    else:

        @click.command()
        @click.option('--attribute', default='Eyeglasses', help='Attribute to visualize')
        @click.option('--filename', default='data/basis10000.json', help='File to load boundary from')
        @click.option('--plot', required=True, default='single',type=click.Choice(['single', 'big']), help='Plot the boundary with a reference image')
        @click.option('--alpha', default=3, type=float, help='Alpha value for single plot')
        @click.option('--auto_alpha', default=False, type=bool, help='Automatic alpha value for single plot')
        @click.option('--seed', default=0, type=int, help='Seed for reference image')
        @click.option('--start',default=-3, type=float, help='Start value for alpha in big plot')
        @click.option('--end',default=3, type=float, help='End value for alpha in big plot')
        @click.option('--noise_mode', default='const', type=click.Choice(['const', 'random', 'none']), help='Noise mode for synthesis')
        @click.option('--save',default=False,type=bool, help='Save the plot as an image')
        @click.option('--proj', default=False, type=bool, help='Project a real image to edit')
        @click.option('--img_proj', default='data/img_proj.png', help='Image to project')
        def main(attribute, filename,plot,alpha,auto_alpha, seed,start,end, noise_mode,save,proj,img_proj):
            """
            Charge la frontière et le centroide pour un attribut donné et génère des images éditées
            """

            # charger la frontière et le centroïde pour l'attribut donné
            boundary = load_boundary(attribute, filename)
            mu_pos = load_mu(attribute, 'data/mu_pos10000.json')

            global ws
            global G

            # charger le générateur StyleGAN2 pré-entrainé sur FFHQ
            with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl') as f:
                G = legacy.load_network_pkl(f)['G_ema'].to('cuda')

            boundary = np.array(boundary)

            ## Reshape les vecteur pour correspondre à l'architecture de styleGAN2
            boundary = np.repeat(boundary[np.newaxis, :], 18, axis=0)  # shape (18, 512)
            mu_pos   = np.repeat(mu_pos  [np.newaxis, :], 18, axis=0)  # shape (18, 512)
            ws = np.repeat(ws[:,np.newaxis, :], 18, axis=1)  # shape (10000,18, 512)

            # si on veut projeter une image réelle à éditer
            w_proj = None #par défaut, pas de projection
            if proj:
                # projetter une image réelle dans l'espace latent W
                runner = CliRunner()
                result = runner.invoke(run_projection, [
                    '--network', 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl',
                    '--target', img_proj,
                    '--outdir', 'out',
                ])

                if result.exit_code != 0:
                    print("Erreur lors de la projection :", result.output)
                else:
                    print("Projection réussie.")
                    w_proj = torch.load('out/projected_w.pkl').cpu().numpy()
                    ref = plt.imread('out/proj.png')

            # sinon on en génère une avec la seed donnée
            else:
                # charger l'image de référence si elle existe et sinon la générer
                try:
                    ref = plt.imread(f'data/images/seed{seed:04d}.png')
                except:
                    w_ref = ws[seed].copy()
                    w_ref = torch.from_numpy(w_ref).to('cuda')

                    with torch.no_grad():
                        ref = G.synthesis(w_ref.unsqueeze(0), noise_mode=noise_mode)
                    ref = (ref.permute(0,2,3,1) * 127.5 + 128).clamp(0,255).to(torch.uint8)[0].cpu().numpy()

            # afficher l'édition de l'image de référence
            if plot=='big':
                alpha_list = np.linspace(start, end, 7)
                big_plot(ws, seed, w_proj, boundary, ref,alpha_list=alpha_list,noise_mode=noise_mode,save=save)
            else: 
                # calculer le pas alpha automatiquement si demandé
                if auto_alpha:
                    alpha = compute_alpha(ws[seed], boundary, mu_pos)
                single_plot(ws, seed, w_proj, boundary, ref, alpha=alpha, noise_mode=noise_mode,save=save)

        main()