import os 
from PIL import Image
import matplotlib.pyplot as plt 

def generate_image_grid(outputs, r=1, c=4, folder='data/image_grid/'):
    """  
    Generate a grid plot from images.
    Saves image to a given folder. 

    Args
      outputs: A list of images
      r: Number of rows.
      c: Number of columns.
      folder: place to save grid to 
    """
    # GENERATES A GRID OF IMAGES
    fig, axes = plt.subplots(r, c, figsize=(25,25))
    for i, ax in enumerate(axes.flat):
        if i < len(outputs):
            ax.imshow(outputs[i])
        ax.set_xticks([])
        ax.set_yticks([])

    # SAVES THE GRID AS PNG TO THE FOLDER
    if not os.path.exists(folder):
        os.makedirs(folder)
    grid_number = len(os.listdir(folder))
    if not os.path.exists(folder + str(grid_number)):
        os.makedirs(folder + str(grid_number))
    fig.savefig(f"{folder + str(grid_number)}/grid.png")

    # SAVES ALL THE INDIVIDUAL IMAGES 
    for i, output in enumerate(outputs):
        output.save(f"{folder + str(grid_number)}/image{i}.png")
    
    # DISPLAYS THE GRID OUTPUT TO SCREEN 
    plt.show()





