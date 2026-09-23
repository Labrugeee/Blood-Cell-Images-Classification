#Exploration DATA
import os
import  cv2# python-m pip install opencv-python
import numpy
import matplotlib.pyplot as plt
import numpy.random as npr
#COUNT PICTURE/GROUPS

#PATH
path_all_pictures = 'D:/Raphael/Documents/PROJET_MLB/data/dataset-master/JPEGImages/'

path_eosinophil = 'D:/Raphael/Documents/PROJET_MLB/data/dataset2-master/dataset2-master/images/TEST/EOSINOPHIL/'

path_lymphocyte = 'D:/Raphael/Documents/PROJET_MLB/data/dataset2-master/dataset2-master/images/TEST/LYMPHOCYTE/'

path_monocyte = 'D:/Raphael/Documents/PROJET_MLB/data/dataset2-master/dataset2-master/images/TEST/MONOCYTE/'

path_neutrophil = 'D:/Raphael/Documents/PROJET_MLB/data/dataset2-master/dataset2-master/images/TEST/NEUTROPHIL/'
commun_path = 'D:/Raphael/Documents/PROJET_MLB/data/dataset2-master/dataset2-master/images/TEST/'
all_path = {
    "EOSINOPHIL": os.listdir(path_eosinophil),
    "LYMPHOCYTE": os.listdir(path_lymphocyte),
    "MONOCYTE": os.listdir(path_monocyte),
    "NEUTROPHIL": os.listdir(path_neutrophil),
}
#1 = EOSINOPHIL
# 2 = LYMPHOCYTE
#3 = MONOCYTE
#4 = NEUTROPHIL
#Count picture/groups and all
def count_all_types(path_all):
    total = 0

    for group, images in path_all.items():
        #print(images)
        
        count = len(images)
        #print(f"{group} : {count}")
        total += count

    #print(f"TOTAL : {total}")

    return total

#count_all_types(all_path)


# Colorimétrie

def color_picture(path_all):
 
    colors_by_group = {} # pour garder les couleurs de chaque image, par groupe
 
    for group, images in path_all.items():
        #print(images)
        colors_by_group[group] = []
 
        for img in images:
            pth = cv2.imread( commun_path + group + '/'+ img)

            colo = numpy.mean(pth,axis=(0,1))
            #print(colo)
            colors_by_group[group].append(colo)
 
    return colors_by_group
 
colors_by_group = color_picture(all_path)
 

# Variance / ecart type par groupe

def variance_picture(colors_by_group):
 
    for group, colors in colors_by_group.items():
        colors = numpy.array(colors) # donne les couelur en 2 dimensions avec une ligne pour uen couelrue et une colone corresponds à un canal
        ecart_type = colors.std(axis=0)
        print(f"{group} : ecart-type (B,V,R) : {ecart_type}")
 
#variance_picture(colors_by_group)
 

#canau = ['Bleu', 'Vert', 'Rouge']

# Graphique avec plt

def plot_variance(colors_by_group):
 
    canaux = ['Bleu', 'Vert', 'Rouge']
 
    fig, axes = plt.subplots(1,3, figsize=(15,5)) # pour avoir 3 graph et pas jsute 1 graph
    #print(fig) # taille de fig
    for i in range(3):
        data = []
        labels = []
        for group, colors in colors_by_group.items():
            colors = numpy.array(colors)
            #print(colors[i])
            #print(group)
            data.append(colors[:, i]) # donc on garde toute les lignes et i pour la couleur car colors est en 2 dimensions
            # en gros con garde juste la colonne i qui corresponde au valeur de bleu vert ou rouge
            # donc prends toutes les couelrus mais garde uniquement la valeur du canal i
            
            labels.append(group)
 
        axes[i].boxplot(data, label=labels)
        axes[i].set_title(canaux[i])
 
    plt.show()
 
#plot_variance(colors_by_group)

#seaborn à voir pour une 
# voir pour faire un nuage de point et je color par type pour voir la distribution des données
# 





# def plot_scatter_rgb(colors_by_group):
#     couleurs_affichage = {"EOSINOPHIL": "orange", "LYMPHOCYTE": "blue",
#                            "MONOCYTE": "green", "NEUTROPHIL": "red"}
#     noms_canaux = ["Bleu", "Vert", "Rouge"]  # ordre BGR d'OpenCV

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     for i in range(3):
#         x_courant = 0  # réinitialisé pour chaque canal
#         for group, colors in colors_by_group.items():
#             colors = numpy.array(colors)
#             valeurs_canal = colors[:, i]

#             n = len(valeurs_canal)
#             x_indices = numpy.arange(x_courant, x_courant + n)

#             axes[i].scatter(x_indices, valeurs_canal, c=couleurs_affichage[group],
#                              label=group, alpha=0.5, s=10)

#             x_courant += n  # décale le point de départ pour le groupe suivant

#         axes[i].set_title(noms_canaux[i])
#         axes[i].set_xlabel("Index image")
#         axes[i].legend()

#     plt.tight_layout()
#     plt.show()

# plot_scatter_rgb(colors_by_group)

def plot_distribution_superposee(colors_by_group):
    couleurs_affichage = {"EOSINOPHIL": "orange", "LYMPHOCYTE": "blue",
                           "MONOCYTE": "green", "NEUTROPHIL": "red"}
    noms_canaux = ["Bleu", "Vert", "Rouge"]  # ordre BGR d'OpenCV

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        for group, colors in colors_by_group.items():
            colors = numpy.array(colors)
            valeurs_canal = colors[:, i]

            y_jitter = npr.normal(0, 0.3, size=len(valeurs_canal))

            axes[i].scatter(valeurs_canal, y_jitter, c=couleurs_affichage[group],
                             label=group, alpha=0.4, s=10)

        axes[i].set_title(f"Distribution du canal {noms_canaux[i]}")
        axes[i].set_xlabel(f"Valeur moyenne {noms_canaux[i]}")
        axes[i].set_yticks([])  # l'axe Y n'a pas de sens réel, juste étalement visuel
        axes[i].legend()

    plt.suptitle("Distribution RVB par type cellulaire — groupes superposés")
    plt.tight_layout()
    plt.show()

plot_distribution_superposee(colors_by_group)