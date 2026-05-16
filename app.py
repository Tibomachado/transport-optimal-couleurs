import time
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import ot


TAILLE_IMAGE = 256

IMAGES_SOURCES = {
    "Bibliothèque Paris 8": "p8_bibliotheque_256.png",
    "Façade Paris 8": "p8_facade_256.png",
    "Passerelle Paris 8": "p8_passerelle_256.png",
}

IMAGES_CIBLES = {
    "Van Gogh - toile complète": "starry_night_256.png",
    "Van Gogh - ciel recadré": "starry_night_skycrop_256.png",
}


def charger_image_depuis_chemin(chemin, taille=TAILLE_IMAGE):
    image = Image.open(chemin).convert("RGB")
    image = image.resize((taille, taille))
    tableau = np.array(image, dtype=float) / 255.0
    return tableau, image


def charger_image_depuis_upload(fichier, taille=TAILLE_IMAGE):
    image = Image.open(fichier).convert("RGB")
    image = image.resize((taille, taille))
    tableau = np.array(image, dtype=float) / 255.0
    return tableau, image


def extraire_palette(image, k):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pixels)

    couleurs = kmeans.cluster_centers_

    compteur = np.bincount(labels, minlength=k).astype(float)
    poids = compteur / compteur.sum()

    return couleurs, poids, labels


def transfert_couleurs(source, cible, k, epsilon, normaliser_cout=True):
    debut = time.perf_counter()

    couleurs_source, poids_source, labels_source = extraire_palette(source, k)
    couleurs_cible, poids_cible, _ = extraire_palette(cible, k)

    C = ot.dist(couleurs_source, couleurs_cible, metric="sqeuclidean")

    if normaliser_cout and C.max() > 0:
        C = C / C.max()

    gamma = ot.sinkhorn(
        poids_source,
        poids_cible,
        C,
        epsilon,
        numItermax=2000,
        stopThr=1e-7,
    )

    nouvelles_couleurs = (gamma @ couleurs_cible) / poids_source[:, None]
    nouvelles_couleurs = np.clip(nouvelles_couleurs, 0, 1)

    pixels_resultat = nouvelles_couleurs[labels_source]
    image_resultat = pixels_resultat.reshape(source.shape)

    temps = time.perf_counter() - debut

    return image_resultat, temps


def convertir_en_image(tableau):
    tableau = np.clip(tableau, 0, 1)
    tableau_uint8 = (tableau * 255).astype(np.uint8)
    return Image.fromarray(tableau_uint8)


def image_vers_buffer(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


st.set_page_config(
    page_title="Transport optimal - Transfert de couleurs",
    layout="wide",
)

st.title("Transport optimal : transfert de couleurs")

st.write(
    "Cette application applique un transfert de couleurs entre deux images "
    "à l'aide de K-Means, de l'algorithme de Sinkhorn et d'une projection barycentrique."
)

st.sidebar.header("Paramètres")

mode = st.sidebar.radio(
    "Choix des images",
    ["Utiliser les images du projet", "Importer mes propres images"],
)

k = st.sidebar.slider(
    "Nombre de couleurs dominantes k",
    min_value=8,
    max_value=256,
    value=96,
    step=8,
)

epsilon = st.sidebar.select_slider(
    "Paramètre entropique ε",
    options=[0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5],
    value=0.03,
)

normaliser_cout = st.sidebar.checkbox(
    "Normaliser la matrice de coût",
    value=True,
)

st.sidebar.info(
    "Petit ε : résultat plus marqué, mais calcul plus fragile.\n\n"
    "Grand ε : résultat plus doux, couleurs plus moyennées."
)

source = None
cible = None
source_pil = None
cible_pil = None

if mode == "Utiliser les images du projet":
    choix_source = st.sidebar.selectbox(
        "Image source",
        list(IMAGES_SOURCES.keys()),
    )

    choix_cible = st.sidebar.selectbox(
        "Image cible",
        list(IMAGES_CIBLES.keys()),
        index=1,
    )

    chemin_source = Path(IMAGES_SOURCES[choix_source])
    chemin_cible = Path(IMAGES_CIBLES[choix_cible])

    if chemin_source.exists() and chemin_cible.exists():
        source, source_pil = charger_image_depuis_chemin(chemin_source)
        cible, cible_pil = charger_image_depuis_chemin(chemin_cible)
    else:
        st.error("Une image du projet est manquante dans le dépôt GitHub.")

else:
    col_upload_1, col_upload_2 = st.columns(2)

    with col_upload_1:
        fichier_source = st.file_uploader(
            "Image source",
            type=["png", "jpg", "jpeg"],
            key="source",
        )

    with col_upload_2:
        fichier_cible = st.file_uploader(
            "Image cible",
            type=["png", "jpg", "jpeg"],
            key="cible",
        )

    if fichier_source is not None and fichier_cible is not None:
        source, source_pil = charger_image_depuis_upload(fichier_source)
        cible, cible_pil = charger_image_depuis_upload(fichier_cible)


if source is not None and cible is not None:
    st.subheader("Images utilisées")

    col1, col2 = st.columns(2)

    with col1:
        st.image(source_pil, caption="Image source", use_container_width=True)

    with col2:
        st.image(cible_pil, caption="Image cible", use_container_width=True)

    lancer = st.button("Lancer le transfert de couleurs")

    if lancer:
        with st.spinner("Calcul du transfert de couleurs en cours..."):
            resultat, temps = transfert_couleurs(
                source,
                cible,
                k,
                epsilon,
                normaliser_cout,
            )

        resultat_pil = convertir_en_image(resultat)

        st.success(f"Calcul terminé en {temps:.2f} secondes")

        st.subheader("Résultat")

        col3, col4, col5 = st.columns(3)

        with col3:
            st.image(source_pil, caption="Source", use_container_width=True)

        with col4:
            st.image(cible_pil, caption="Cible", use_container_width=True)

        with col5:
            st.image(resultat_pil, caption="Image recolorée", use_container_width=True)

        st.write("Paramètres utilisés :")
        st.write(f"- k = {k}")
        st.write(f"- ε = {epsilon}")
        st.write(f"- normalisation de la matrice de coût = {normaliser_cout}")

        buffer = image_vers_buffer(resultat_pil)

        st.download_button(
            label="Télécharger l'image résultat",
            data=buffer,
            file_name=f"resultat_k{k}_epsilon{str(epsilon).replace('.', '')}.png",
            mime="image/png",
        )

else:
    st.warning("Choisis les images du projet ou importe une image source et une image cible.")
