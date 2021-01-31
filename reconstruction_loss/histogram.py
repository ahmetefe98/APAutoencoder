import json
import matplotlib.pyplot as plt
import numpy as np

with open('./autoencoder/ae_list_of_norm.txt', 'r') as f:
    ae_list_of_norm = json.load(f)

with open('./pca/pca_list_of_norm.txt', 'r') as f:
    pca_list_of_norm = json.load(f)

bins = np.linspace(0,500,200)
plt.hist(ae_list_of_norm, bins, alpha=0.5, density=True, label="Autoencoder")
plt.hist(pca_list_of_norm, bins, alpha=0.5, density=True, label="PCA")
plt.legend(loc='upper right')
plt.title("Rekonstruktionsfehler Histogramm")
plt.xlabel("Summe der absoluten Differenzen")
plt.ylabel("Anteil an der Gesamtanzahl")
plt.show()