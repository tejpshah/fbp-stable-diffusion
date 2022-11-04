import torch 
from torch import autocast 
from tqdm.auto import tqdm 
from PIL import Image 
from transformers import CLIPTextModel, CLIPTokenizer 
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler 

device = "cuda"

"""
GENERAL IDEA BEHIND STABLE DIFFUSION (INFERENCE COMPONENT):

STEP 1: Take as input a text prompt and convert it into a latent vector. 
CLIP_TEXT_ENCODER(TEXT PROMPT) = TEXT EMBEDDINGS. 

STEP 2: In The backward diffusion process, predict the preivous noisy image i-1 from current noisy image i using UNET. 
OPTIONLLY, YOU CAN ALSO CONDITION ON OTHER INFORMATION SUCH AS TEXT, IMAGES, ETC TO GUIDE THE DIFFUSION PROCESS (i.e. TEXT EMBEDDINGS) 
UNET(FORWARD_LATENT_j) ~= FORWARD_LATENT_{j-1} for j = n, n-1, ..., 1

STEP 3: After the backward diffusion process, convert the processed latent vector at the end of backward diffusion using the decoder.
GENERATED_IMAGES = VAE.DECODE()
"""

# Load Pre-Trained Tokenizer (CLIP : Contrastive Language Image Pre-Training) and Encoder
# tokenizer(prompt) = [...,...,...] consists of atomic tokens 
# text_encoder(tokenizer(prompt)) = [[..,...], [...,...], [...,...]] converts each atomic token into embeddings
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = text_encoder.to(device)

# Loads Pre-Trained U-Net to support reverse diffusion process, to help predict image image_t -> image_{t-1} for n steps
# unet(latent_n_forward_process) = latent_1_backward_process
unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=True)
unet = unet.to(device)

# Load Pre-Trained VAE which can encode and decode images from latents
# vae.encode(image) = latent, vae.decode(latent) = image
vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=True)
vae =vae.to(device)

# Load a scheduler for doing inference with the pre-trained models
# During training, the forward and backward diffusion process had n=1000 steps
# During inference, we want to get results faster so this makes the forward and 
# backward procession converge to Gausian noise and relevant information respectively. 
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000)

def encode_text(prompt):

    # CONDITIONED: Generates embeddings for the text prompt
    tokenized_text_conditioned = tokenizer(prompt, padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    with torch.no_grad(): tokenized_text_conditioned = text_encoder(tokenized_text_conditioned.input_ids.to(device))[0]

    # UNCONDITIONED: Generates embeedings for empty strings the size of the text prompt 
    tokenized_text_unconditioned = tokenizer([''] * len(prompt), padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
    with torch.no_grad(): tokenized_text_unconditioned = text_encoder(tokenized_text_unconditioned.input_ids.to(device))[0]

    # concatenates unconditioned and conditioned embddings and returns the resulting vector
    embeddings = torch.cat([tokenized_text_unconditioned, tokenized_text_conditioned])
    return embeddings  

def reverse_diffusion_process(text_embeddings, height=512, width=512, time=50, guidance_scale=7.5, latents=None):

    # generates latent vector if it doesn't exist and sends it to device
    if latents is None: latents = torch.randn(text_embeddings.shape[0] // 2, unet.in_channels, height//8, width//8)
    latents = latents.to(device)

    # initializes scheduler and first latent at t_n with some noise 
    scheduler.set_timesteps(time)
    latents *= scheduler.sigmas[0]       
    
    # use autocast to dynamically choose dtypes for mixed precision
    with autocast(device): 
        for i, t in tqdm(enumerate(scheduler.timesteps)):

            # concatenate unconditioned and conditioned vector
            combined = torch.cat([latents]*2)

            # peturb the combined by some noise (determined by scheduler)
            sigma = scheduler.sigmas[i]
            combined = combined / ((1+sigma**2) ** 0.5)

            # predict the vector from the prevoius timestep
            with torch.no_grad(): z = unet(combined, t, encoder_hidden_states=text_embeddings)['sample']
            z_pred_unconditioned, z_pred_conditioned = z.chunk(2) 

            # use the conditioned vector to perform guidance on diffusion process
            z = z_pred_unconditioned + guidance_scale * (z_pred_conditioned - z_pred_unconditioned)

            # determine the noisy sample with the guidance
            latents = scheduler.step(z, i, latents)['prev_sample']

    return latents 

def decode_latent(latents, beta=0.18215):

    # scale the latents by a factor, found 0.18215 as a good parameter online
    latents = latents / beta 

    # decode the latent vector representation into image outputs 
    with torch.no_grad(): images = vae.decode(latents)

    # normalize and convert the outputs into a PIL image
    images = (images/2 + 0.5).clamp(0,1)
    images = images.detach().cpu().permute(0,2,3,1).numpy() 
    images = (images*255).round().astype('uint8')

    # return the decoded images
    decoded = [Image.fromarray(image) for image in images]
    
    return decoded 

def inference(prompts, height=512, width=512, time=50, guidance_scale=7.5, latents=None):

    # since we don't have many gpus, we run each prompt individually 
    generations = [] 

    # automatically cast the prompts to array 
    if type(prompts) == str: prompts = [prompts]

    for i in range(len(prompts)):

        # encode text prompt to guide diffusion process from Gaussian Noise
        text_embedding = encode_text(prompts[i])

        # resultant vector after UNET backward diffusion process with CLIP Guided Text Diffusion 
        final_latent = reverse_diffusion_process(text_embedding, height=height, width=width, time=time, guidance_scale=guidance_scale, latents=latents)
        
        # use VAE to decode final latents to generate the image outputs 
        decoded_outputs = decode_latent(final_latent)

        # add generated images to all the generated images that we have 
        generations.append(decoded_outputs)

    # return the generated images after inference
    return generations

