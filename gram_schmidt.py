import numpy as np
import json

# Fonction pour effectuer l'orthogonalisation de Gram-Schmidt sur une liste de vecteurs
def gm(vectors):
    N = len(vectors)
    orthogonal_vectors = []
    v = vectors[0]
    # Normalisation du premier vecteur
    orthogonal_vectors.append(v / np.linalg.norm(v))
    for i in range(1, N):
        v = vectors[i]
        # Soustraction des projections sur les vecteurs déjà orthogonalisés
        for j in range(i):
            v -= np.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        # Ajout du vecteur normalisé s'il n'est pas nul
        if np.linalg.norm(v) > 1e-10:
            orthogonal_vectors.append(v / np.linalg.norm(v))
    return np.array(orthogonal_vectors)

if __name__ == "__main__":
    # Chargement des frontières depuis un fichier JSON
    with open('data/boundary10000.json', 'r') as f:
        boundaries = json.load(f)

    vectors = []
    # Conversion des frontières en tableaux numpy
    for attribute in boundaries.keys():
        boundary = np.array(boundaries[attribute])
        vectors.append(boundary)
    
    # Application de Gram-Schmidt pour obtenir des vecteurs orthogonaux
    orthogonal_boundaries = gm(vectors)

    # Conversion en format compatible JSON
    orthogonal_boundaries = orthogonal_boundaries.tolist()
    orthogonal_boundaries = {attribute: boundary for attribute, boundary in zip(boundaries.keys(), orthogonal_boundaries)}

    # Sauvegarde des vecteurs orthogonaux dans un nouveau fichier JSON
    with open('data/basis10000.json', 'w') as f:
        json.dump(orthogonal_boundaries, f, indent=2)