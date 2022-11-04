import model 
import utils 
import torch 

# GENERATES N NUMBER OF IMAGES WITH A GIVEN PROMPT
NUM_GENERATIONS = 1
prompt = "Steampunk ship flying through the clouds, digital art"

# GENERATES N IMAGES FROM THE GIVEN PROMPT BASED ON SOME RANDOM Z
z = torch.randn((1, model.unet.in_channels, 512 // 8, 512 // 8))
OriginalImages = model.inference([prompt] * NUM_GENERATIONS, time=20, latents=z)[0] 

# GENERATES A VARIATION OF THE IMAGE PROMPT GIVEN THE RANDOM Z PETURBED
z_new = model.decode_variations(z, scale=0.2)
NewImage = model.inference([prompt] * NUM_GENERATIONS, time=20, latents=z_new)[0] 