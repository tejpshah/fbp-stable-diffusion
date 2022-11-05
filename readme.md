# Foundations of Business Programming | F22 Tej Shah
This final project provided me practical exposure to implementing state of the art pre-trained models into practical application.
In the last several months, there is a surge in popularity surrounding Generative AI with the rise of notable image generation
models such as DALLE-2, Imagen, and more. Stable Diffusion is an open-source version of image generation which has so far 
achieved impressive results. 

For this project, you take as input text prompts in `prompts.txt` which yields the following outputs:
(1) 2 image generations; (2) 2 variations of the images from (1); (3) video generation of diffusion process for all images.

The output of sample generations from  `prompts.txt` can be seen here [here](data/). 

The github repository is here: https://github.com/tejpshah/fbp-stable-diffusion

To make each of these possible, I detail the main ideas below:

## (1) Image Generation:
Stable Diffusion is trained on a vast variety of images from the Internet. As part of the diffusion training process, there
is a forward process and a backward process. For the forward process, random noise is added at each timestep for 1000 steps
until the uncorrupted image is completely destroyed (completely Gausian noise). At that point, the backward process is trained
on the objective to reconstruct the less uncorrupted from the t-1th timestep given the context of the corrupted image at the tth timestep.
For efficiency and better representations, this forward and backward diffusion process happens in the latent space. Naively, to generate
an image from Gaussian noise enables any image from the distribution of images stable diffusion is trained on to be generated. To guide
the diffusion process and reduce the search space, we can concatenate other representations during inference in the backward diffusion process.
Essentially, we can concatentate a good textual representation from SOTA models like OpenAI's CLIP (which jointly align text and images) to our 
pre-existing intial latent vector from which we start the backward diffusion process. At the very end of our diffusion process, we have a generated
latent that can be decoded using a trained Variational Autoencoder (VAE) module. 

## (2) Image Variations:
To generate image variations, crucially realize that the diffusion process is a Markov Chain, which means it obeys the Markov Property: 
the current output only depends on the previous output. Hence, if we have similar starting states from the vector representation, we can
generate similar variations of an image. To do so, I peturb the initial starting latent with some slight noise before being guided by other representations
like CLIP. 

## (3) Video Generation:
To generate video generations, simply store the latent at each time step of the diffusion process and then decode each latent to a frame and create a video
that generates the video from all the frames that are decoded. 

## (4) Takeaways 
It was interesting to learn how to use different pre-trained modules like UNET, CLIP, and VAE to do inference. 
Building out the pipeline for image-inference from scratch crucially enables you to build other interesting features like image-variations or video-generation.
I found it challenging to run it on the `ilab` machines since enough resources were not allocated to download the models to perform inference. 
Otherwise, if I had enough resources, I would have explored looking into splitting image generation on to multiple GPUs to speed up inference time. 
Instead, I decided to run my scripts on a Google Colab environment so I would have access to a GPU for image generation. 
After generating the first 5 text prompts, I use GPT-3 to generate me another 42 text prompts. 