import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#PATH
# path_all_pictures = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN'

path_eosinophil = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/EOSINOPHIL/'

path_lymphocyte = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/LYMPHOCYTE/'

path_monocyte = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/MONOCYTE/'

path_neutrophil = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/NEUTROPHIL/'

# path_Rawdata = "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/RAW/"

def import_images(path):
    """Read images from a folder and made a dictionnary of arrays

    Args:
        path (path): path to raw data folder
    """
    images = []
    with os.scandir(path) as entries:
        for i,entry in enumerate(entries):
            px_matrix = cv2.imread(entry.path)
            px_matrix = remove_black_borders(px_matrix)
            px_matrix = cv2.resize(px_matrix, (320, 240))
            images.append([i, px_matrix])
    return images

def remove_black_borders(img, threshold=30):
    """
    Supprime les coins noirs d'une image rectangulaire inclinée.
    """
    # Passage temporaire en grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convertir en niveaux de gris car couleur pas nécessaire pour détecter les contours

    # Les pixels suffisamment clairs correspondent à l'image utile
    mask = (gray > threshold).astype(np.uint8) * 255 # creation d'un masque binaire où les pixels clairs sont blancs (255) et les pixels sombres sont noirs (0) et astype(np.uint8) pour que le masque soit compris en 0 pour False et donc noir et 255 pour True et donc blanc

    # Fermer les petits trous
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # morph.close permet de fermer les petits trous dans le masque binaire, en utilisant un noyau de 7x7 pixels étant la taille du noyau utilisée. Cela aide à obtenir une forme plus continue pour l'image utile. Parce que entre les pixels clairs se trouvent des pixels sombres, ce qui peut créer des trous dans le masque. En fermant ces trous, on obtient une forme plus continue pour l'image utile.

    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # On recherche les contours externes de l'image utile dans le masque binaire. cv2.RETR_EXTERNAL permet de ne récupérer que les contours externes, et cv2.CHAIN_APPROX_SIMPLE permet de simplifier les contours en ne conservant que les points essentiels.

    if not contours:
        return img # permet l'image originale si aucun contour n'a été trouvé

    # Plus grand contour = zone de l'image
    contour = max(contours, key=cv2.contourArea) # permet de trouver le contour ayant la plus grande aire, qui correspond à la zone de l'image utile.
    # Périmètre du contour
    perimeter = cv2.arcLength(contour, True) # permet de calculer le périmètre du contour trouvé, ce qui est utile pour l'approximation des coins.
    
    # Chercher une approximation avec 4 coins
    points = None
    for epsilon_ratio in np.arange(0.001, 0.1, 0.001):
        epsilon = epsilon_ratio * perimeter
        corners = cv2.approxPolyDP(contour, epsilon, True)

        if len(corners) == 4:
            points = corners.reshape(4, 2).astype(np.float32)
            break

    # Sécurité
    if points is None:
        print("Impossible de trouver 4 coins")
        return img
    
    # --------------------------------------------------------
    # Ordonner les coins :
    # haut-gauche
    # haut-droit
    # bas-droit
    # bas-gauche
    # --------------------------------------------------------

    ordered = np.zeros((4, 2), dtype=np.float32)

    somme = points.sum(axis=1)
    difference = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(somme)]       # haut gauche
    ordered[2] = points[np.argmax(somme)]       # bas droite
    ordered[1] = points[np.argmin(difference)]  # haut droite
    ordered[3] = points[np.argmax(difference)]  # bas gauche

    # --------------------------------------------------------
    # Transformation perspective
    # --------------------------------------------------------

    target_width = 320
    target_height = 240

    destination = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered, destination)

    # Appliquer la transformation
    result = cv2.warpPerspective(img,matrix,(target_width, target_height))

    return result


def show_img(img):
    # BGR -> RGB pour matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

img_easinophil = import_images(path_eosinophil)
print(len(img_easinophil))
img_lymphocyte = import_images(path_lymphocyte)
print(len(img_lymphocyte))
img_monocyte = import_images(path_monocyte)
print(len(img_monocyte))
img_neutrophil = import_images(path_neutrophil)
print(len(img_neutrophil))
show_img(img_easinophil[0][1])
show_img(img_lymphocyte[0][1])
show_img(img_monocyte[0][1])
show_img(img_neutrophil[0][1])

all_img = {"eosinophil": img_easinophil, "lymphocyte": img_lymphocyte, "monocyte": img_monocyte, "neutrophil": img_neutrophil}

#