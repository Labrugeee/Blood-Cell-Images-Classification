#Exploration DATA
import os
import  cv2# python-m pip install opencv-python
import numpy
import matplotlib.pyplot as plt
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
 
    fig, axes = plt.subplots(1, 3, figsize=(15,5)) # pour avoir 3 graph et pas jsute 1 graph
    #print(fig) # taille de fig
    for i in range(3):
        data = []
        labels = []
        for group, colors in colors_by_group.items():
            colors = numpy.array(colors)
            print(colors[i])
            #print(group)
            data.append(colors[:, i]) # donc on garde toute les lignes et i pour la couleur car colors est en 2 dimensions
            # en gros con garde juste la colonne i qui corresponde au valeur de bleu vert ou rouge
            # donc prends toutes les couelrus mais garde uniquement la valeur du canal i
            labels.append(group)
 
        axes[i].boxplot(data, label=labels)
        axes[i].set_title(canaux[i])
 
    plt.show()
 
plot_variance(colors_by_group)

#seaborn à voir pour une 
# voir pour faire un nuage de point et je color par type pour voir la distribution des données
# 






# Graphique avec plt 
 
def plot_scatter(colors_by_group):
 
    canaux = ['Bleu', 'Vert', 'Rouge']
 
    fig, axes = plt.scatter(1, 3) # pour avoir 3 graph et pas jsute 1 graph
    #print(fig) # taille de fig
    for i in range(3):
        data = []
        labels = []
        for group, colors in colors_by_group.items():
            colors = numpy.array(colors)
            print(colors[i])
            #print(group)
            data.append(colors[:, i]) # donc on garde toute les lignes et i pour la couleur car colors est en 2 dimensions
            # en gros con garde juste la colonne i qui corresponde au valeur de bleu vert ou rouge
            # donc prends toutes les couelrus mais garde uniquement la valeur du canal i
            labels.append(group)
 
        axes[i].boxplot(data, label=labels)
        axes[i].set_title(canaux[i])
 
    plt.show()
 
plot_scatter(colors_by_group)
