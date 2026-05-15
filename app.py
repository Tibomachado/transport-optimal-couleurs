import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import ot


# Paramètres principaux de l'expérience
TAILLE_IMAGE = 256
K = 96
REG = 0.03

# Images sources
EXPERIENCES = [
    ("p8_bibliotheque_256.png", "bibliotheque"),
    ("p8_facade_256.png", "facade"),
    ("p8_passerelle_256.png", "passerelle"),
]

# Images cibles
CIBLES = [
    ("full", "starry_night_256.png"),
    ("skycrop", "starry_night_skycrop_256.png"),
]


def formater_reg(reg):
    return str(reg).replace(".", "")


def creer_dossier_resultat():
    reg_txt = formater_reg(REG)
    base_nom = f"Result_k{K}_reg{reg_txt}"

    numero = 1
    while Path(f"{base_nom}_{numero}").exists():
        numero += 1

    dossier = Path(f"{base_nom}_{numero}")
    dossier.mkdir()

    return dossier


def construire_nom_resultat(nom_source, nom_cible):
    reg_txt = formater_reg(REG)
    return f"result_{nom_source}_target-{nom_cible}_k{K}_reg{reg_txt}.png"


def charger_image(nom_fichier):
    image = Image.open(nom_fichier).convert("RGB")
    image = image.resize((TAILLE_IMAGE, TAILLE_IMAGE))

    return np.array(image, dtype=float) / 255.0


def extraire_palette(image, k):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pixels)

    couleurs = kmeans.cluster_centers_

    compteur = np.bincount(labels, minlength=k).astype(float)
    poids = compteur / compteur.sum()

    return couleurs, poids, labels


def transfert_couleurs(image_source_path, image_cible_path, chemin_resultat):
    print(f"\nTraitement de {image_source_path} vers {chemin_resultat}")
    debut = time.perf_counter()

    source = charger_image(image_source_path)
    cible = charger_image(image_cible_path)

    couleurs_source, poids_source, labels_source = extraire_palette(source, K)
    couleurs_cible, poids_cible, _ = extraire_palette(cible, K)

    # Matrice de coût : C_ij = ||x_i - y_j||²
    C = ot.dist(couleurs_source, couleurs_cible, metric="sqeuclidean")

    # Plan de transport régularisé
    gamma = ot.sinkhorn(poids_source, poids_cible, C, REG)

    # Projection barycentrique
    nouvelles_couleurs = (gamma @ couleurs_cible) / poids_source[:, None]
    nouvelles_couleurs = np.clip(nouvelles_couleurs, 0, 1)

    pixels_resultat = nouvelles_couleurs[labels_source]
    image_resultat = pixels_resultat.reshape(source.shape)

    image_resultat = (image_resultat * 255).astype(np.uint8)
    Image.fromarray(image_resultat).save(chemin_resultat)

    fin = time.perf_counter()
    print(f"Image créée : {chemin_resultat}")
    print(f"Temps de calcul : {fin - debut:.2f} secondes")


def main():
    print("=== Transfert de couleurs avec transport optimal ===")
    print(f"k = {K}")
    print(f"reg = {REG}")

    dossier_resultat = creer_dossier_resultat()
    print(f"Dossier créé : {dossier_resultat}")

    for image_source, nom_source in EXPERIENCES:
        for nom_cible, image_cible in CIBLES:
            nom_resultat = construire_nom_resultat(nom_source, nom_cible)
            chemin_resultat = dossier_resultat / nom_resultat

            transfert_couleurs(
                image_source_path=image_source,
                image_cible_path=image_cible,
                chemin_resultat=chemin_resultat
            )

    print("\nTerminé : tous les résultats ont été créés.")
    print(f"Ils se trouvent dans le dossier : {dossier_resultat}")


if __name__ == "__main__":
    main()