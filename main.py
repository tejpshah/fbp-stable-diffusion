import model 
import utils 
import torch 

FILENAME = 'prompts.txt'

with open(FILENAME, 'r') as f:
    for sentence in f:
        prompt = sentence[0:len(sentence)-1]

        # GENERATES N NUMBER OF IMAGES WITH A GIVEN PROMPT
        IMAGES, VIDEOS = [], []
        NUM_GENERATIONS = 2 

        for i in range(NUM_GENERATIONS):    
            # GENERATES N IMAGES FROM THE GIVEN PROMPT BASED ON SOME RANDOM Z
            z = torch.randn((1, model.unet.in_channels, 512 // 8, 512 // 8))
            OriginalImages = model.inference([prompt], time=35, latents=z, generate_video=True)
            utils.generate_video(OriginalImages)

            # GENERATES A VARIATION OF THE IMAGE PROMPT GIVEN THE RANDOM Z PETURBED
            z_new = model.decode_variations(z, scale=0.2)
            NewImages = model.inference([prompt], time=35, latents=z_new, generate_video=True)
            utils.generate_video(NewImages)

            # APPENDS GENERATED AND VARIANT IMAGE TO THE PHOTO
            IMAGES.append(OriginalImages[-1])
            IMAGES.append(NewImages[-1])
            
        utils.generate_image_grid(outputs=IMAGES, r=1, c=4, folder="data/image_grid/")

        



