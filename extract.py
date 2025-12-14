import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data/BAHAMAS/bahamas_1.pkl', 'rb') as file:
    labels, images = pickle.load(file)

    img = images[0]
    print(img.shape)
    plt.imsave('figures/total_mass.png', img[0], cmap='viridis')
    plt.imsave('figures/x_ray.png', img[1], cmap='viridis')
    plt.imsave('figures/stellar_mass.png', img[2], cmap='viridis')
    
    