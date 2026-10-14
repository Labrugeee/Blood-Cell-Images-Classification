# Making classification model for blood cells
# LOUTY Florian
# M1 Bioinformatics - Univeristy of Rennes
# Creation date : 01/10/2026
# Uptdate date : 14/10/2026

# Loading libraries

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Setup paths :

path_all_train = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN'

path_all_val = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TEST/'

path_all_val_simple = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TEST_SIMPLE/'

path_eosinophil = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/EOSINOPHIL/'

path_lymphocyte = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/LYMPHOCYTE/'

path_monocyte = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/MONOCYTE/'

path_neutrophil = 'C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/NEUTROPHIL/'

# path_Rawdata = "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/RAW/"

# Import images

def import_images_for_random_forest(path, size=(320, 240), rmBlackBorders=False):
    images = []
    for fname in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, fname))
        if img is None:          # fichier non lisible
            continue
        if rmBlackBorders:
            img = remove_black_borders(img)
        img = cv2.resize(img, size)
        images.append(img)
    return np.array(images)

def import_images_for_CNN(path):
    """Read images from a folder and made a dictionnary of arrays

    Args:
        path (path): path to raw data folder
    """
    images = []
    with os.scandir(path) as entries:
        for i,entry in enumerate(entries):
            px_matrix = cv2.imread(entry.path)
            # px_matrix = remove_black_borders(px_matrix)
            px_matrix = cv2.resize(px_matrix, (320, 240))
            images.append([i, px_matrix])
    return images


def show_img(img):
    # BGR -> RGB pour matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

# Approach 1 : Classification made by random forest model without editing images

paths_train = {
    "eosinophil": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/EOSINOPHIL/",
    "lymphocyte": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/LYMPHOCYTE/",
    "monocyte": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/MONOCYTE/",
    "neutrophil": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TRAIN/NEUTROPHIL/",
}

paths_test = {
    "eosinophil": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TEST_SIMPLE/EOSINOPHIL/",
    "lymphocyte": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TEST_SIMPLE/LYMPHOCYTE/",
    "monocyte": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TEST_SIMPLE/MONOCYTE/",
    "neutrophil": "C:/Users/flori/OneDrive/Documents/Formations/Bioinfo_Rennes/M2_Bioinfo/MLB/Blood-Cell-Images-Classification/data/TEST_SIMPLE/NEUTROPHIL/",
}

def load_from_paths_for_random_forest(paths, size=(64, 48), flatten=True, rmBlackBorders=False):
    """paths : dict {nom_de_classe: chemin_du_dossier}"""
    class_names = list(paths.keys())
    X, y = [], []
    for label, cls in enumerate(class_names):
        imgs = import_images_for_random_forest(paths[cls], size=size, rmBlackBorders=rmBlackBorders)
        print(f"{cls} : {len(imgs)} images")
        X.append(imgs)
        y.extend([label] * len(imgs))
    X = np.concatenate(X).astype(np.float32) / 255.0
    if flatten:
        X = X.reshape(len(X), -1)
    return X, np.array(y), class_names

def make_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Approach 2 : Classification made by random forest model with editing images

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

# Approach 3 : Classificiation made with CNN model

def train_cnn(img_size=(320, 240),batch_size=32,epochs=5,
               path_train=path_all_train,
               path_val=path_all_val_simple):
    
    train_data = tf.keras.utils.image_dataset_from_directory(path_train, image_size=img_size, batch_size=batch_size,label_mode='categorical')

    val_data = tf.keras.utils.image_dataset_from_directory(path_val, image_size=img_size, batch_size=batch_size,label_mode='categorical')

    # Normaliztion : [0,255] -> [0,1]
    normalization_layer = layers.Rescaling(1./255)

    # Creating model :
    model = models.Sequential([
        normalization_layer,
        layers.Conv2D(10, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(20, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(15, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),

        # Out
        layers.Dense(4, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                    metrics=['accuracy'])

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs)
    
    return model, history


if __name__ == "__main__":
    # For random forest model :
    print("FOR RANDOM FOREST MODEL WITHOUT BLACK BORDERS :")

    # Chargement des données d'entraînement et de test
    X_train, y_train, classes_train = load_from_paths_for_random_forest(paths_train, flatten=True)
    print(X_train.shape, np.bincount(y_train))

    X_test, y_test, classes_test = load_from_paths_for_random_forest(paths_test, flatten=True)
    print(X_test.shape, np.bincount(y_test))

    # Création et entraînement du modèle Random Forest
    rf_model = make_random_forest(X_train, y_train, n_estimators=100, max_depth=None)

    # Prédiction sur les données de test
    y_pred = rf_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=classes_test))

    print("FOR RANDOM FOREST MODEL WITH BLACK BORDERS :")
    # Chargement des données d'entraînement et de test
    X_train_wBB, y_train_wBB, classes_train_wBB = load_from_paths_for_random_forest(paths_train, flatten=True, rmBlackBorders=True)
    print(X_train_wBB.shape, np.bincount(y_train_wBB))

    X_test_wBB, y_test_wBB, classes_test_wBB = load_from_paths_for_random_forest(paths_test, flatten=True, rmBlackBorders=True)
    print(X_test_wBB.shape, np.bincount(y_test_wBB))

    # Création et entraînement du modèle Random Forest
    rf_model_wBB = make_random_forest(X_train_wBB, y_train_wBB, n_estimators=100, max_depth=None)

    # Prédiction sur les données de test
    y_pred_wBB = rf_model_wBB.predict(X_test_wBB)
    print("Accuracy:", accuracy_score(y_test_wBB, y_pred_wBB))
    print("Classification Report:\n", classification_report(y_test_wBB, y_pred_wBB, target_names=classes_test_wBB))

    # Confusion matrix :
    cm_wBB = confusion_matrix(y_test_wBB, y_pred_wBB)
    disp_wBB = ConfusionMatrixDisplay(confusion_matrix=cm_wBB, display_labels=classes_test_wBB)
    disp_wBB.plot()
    plt.show()

    # # For CNN Model :
    # print("FOR CNN Model :")
    # model, history = train_cnn()
    # # Accuracy
    # plt.figure(figsize=(10, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='Train')
    # plt.plot(history.history['val_accuracy'], label='Validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy')
    # plt.legend()
    # plt.grid()

    # # Loss
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Train')
    # plt.plot(history.history['val_loss'], label='Validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss')
    # plt.legend()
    # plt.grid()

    # plt.tight_layout()
    # plt.show()

# img_easinophil = import_images(path_eosinophil)
# print(len(img_easinophil))
# img_lymphocyte = import_images(path_lymphocyte)
# print(len(img_lymphocyte))
# img_monocyte = import_images(path_monocyte)
# print(len(img_monocyte))
# img_neutrophil = import_images(path_neutrophil)
# print(len(img_neutrophil))
# show_img(img_easinophil[0][1])
# show_img(img_lymphocyte[0][1])
# show_img(img_monocyte[0][1])
# show_img(img_neutrophil[0][1])

# all_img = {"eosinophil": img_easinophil, "lymphocyte": img_lymphocyte, "monocyte": img_monocyte, "neutrophil": img_neutrophil}

