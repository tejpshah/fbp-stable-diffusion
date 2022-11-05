import os 
import cv2 
import numpy as np 
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

def generate_video(frames, folder="/data/videos/", fps=10):

    # generates videos folders if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    video_number = len(os.listdir(folder))

    # this genreates the size of the video
    size = (frames[0].width, frames[0].height)

    # this lets you generate the video 
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
    save_at = folder + str(video_number) + ".mp4"

    print(save_at)
    video = cv2.VideoWriter(save_at, fourcc, fps, size)

    # this adds each frame to the video and writes with cv2
    for frame in frames:
        tmp_img = frame.copy()
        video.write(cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR))
    
    # this releases the video from cv2
    video.release()