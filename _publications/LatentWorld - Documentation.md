---
tags:
  - AI
  - Research
  - MA
  - TH_OHM
  - computer_science
---

> [!Moritz Anweisung]
> ### Project documentation
> 
> Alongside the presentation, you are asked to submit a short written documentation of your project:
> 
> **Hand in date: 20.02.2026 via E-Mail**
> **DIN A4 format, 5–10 pages, PDF**

**10 pages** are around **3000 words**

---


**Dieses Paper ist eine gute Vorlage:** 
	https://shreyansh26.github.io/post/2023-03-26_flash-attention/

![[featured.png]]

---

**Diese Struktur wäre ganz gut so:**
### Overview
### Background
### Experiments

---

Du könntest auch so ein Poster kreieren und für den Raum ausdrucken.

![[du könntest ein poster kreieren.png|400]]

---
### Gedanken über das Paper

Schreibe es wie ein Medium Artikel. Ich denke das ist die richtige Höhe.



---


> Ludwig Boltzmann, who spent much of his life studying statistical mechanics, died in 1906, by his own hand. Paul Ehrenfest, carrying on the work, died similarly in 1933. Now it is our turn to study statistical mechanics. 
> 
> ― David L. Goodstein, [States of Matter](https://www.goodreads.com/work/quotes/44809168)


## Hessian Paper Overview

This project is an extrusion of the paper [[Hessian Geometry of Latent Space in Generative Models.pdf]] which proposes a way to explore the Latent Space like you would explore a country. By scattering a two-dimensional Grid of [[Latent|Latents]] the authors analyzed the geometry of the space, revealing phase-transitions between semantic concepts like "cat" or "mountain", that revealed rifts in the geometry where the [[Diffusion|Diffusion Model]] (Stable Diffusion 1.5) got unstable generating either cat or mountain in a fractal pattern, which reached all the way down to the bit-level resolution of the Model. In other parts of the rift there was a third concept appearing during the interpolation of the two concepts. Suddenly a car was appearing.

They achieved this with minimal guidance, using [[Stable Diffusion|Stable Diffusion 1.5]] with prompts: “High quality picture, 4k, detailed” and negative prompts: “blurry, ugly, stock photo” revealing the pure geometry of the space in an unaltered state. The images that occurred (cat, mountain) were an expression of the inherent bias of the models training data.

> the method reveals a fractal structure of phase transitions in the latent space, characterized by abrupt changes in the Fisher metric. 
> 
> [[Hessian Geometry of Latent Space in Generative Models]]

![[hessian_paper_phase_transition.png]]

## Implications (rift between concepts)

The implications of that paper are of course that there is no clear semantic bridge between concepts like "cat" or "mountain" in our world of images and language with which we categorize the world.

Because the semantic landscape of human-constructed words and images is by its definition an imperfect map cast over the territory which psychoanalyst Jacques Lacan would describe as [[Das „Reale“ (Lacan)|the Real]], the indissoluble rest, which cannot be expressed in language or imagination.

It is therefore logical that a Neural Network sampling from a big dataset of our real word concepts and categorizing them on features and semantic distance would inevitably sketch out a similar map again.


![[semantic_landscape.png|300]]

If we would depict this space in geography terms, it would be an archipelago of semi-disconnected islands of varying sizes.

## My Interest for Project

While the goal of the authors why they try to depict this geometry was to find coherent paths through these phase transitions so to guide interpolation between images sensibly.

What peaked my interest in this was taking an effort to explore these semantic rifts that they describe in an effort to find new and unusual images.

It seems at least to me so, that the most interesting output of generative AI models happens when they essentially "fail" at what they do. When there are weird glitches or anomalies that make the output interesting in the same way an artists brushstroke is essentially an alteration of the thing they try to depict, but in this alteration, they reveal something about the medium and the process used to create this depiction. "Neural blobs" or "hallucinations" of a generative model are depictions of the medium akin to "light leaks" or "grain" are for photography, which are flaws that became stylistic elements that artists seek out.

![[neural poster keingarten.jpg|300]]


## Replicating the Paper Setup

So I started to replicate the setup in the paper, to see if these fractals can be replicated. 

#### Paper Setup

The authors used **StableDiffusion 1.5** for their setup with Lykons **Dreamshaper 8** checkpoint loaded. The **positive prompts** applied are “High quality picture, 4k, detailed” with **negative prompts** being “blurry, ugly, stock photo.” The **DDIM parameter** is set to **η = 0**, so no additional noise is added in the generation, which makes the pipeline deterministic and its output reproducible. I additionally used a fixed seed for the generator in the pipeline for every image generated. They used a [[Classifier-free guidance (CFG)]] scale of 5 with 50 inference steps.


> [!Perplexity]
> #### Conda-Environment vs. Kernel
> 
> - **Conda-Environment**: Isolierte Python-Umgebung mit Paketen (PyTorch CUDA, etc.) – dein "gpu"-Env mit RTX 4090-Support.
>     
> - **Kernel**: Registrierter Einstiegspunkt (via `ipykernel install --name=gpu`), der **auf** dem Environment basiert. Zeigt als "Python [conda env:gpu]" in der Kernel-Liste.

#### Setup in Jupyter Lab / My PC

The setup was created in Jupyter Lab.

I created a **Conda-Environment** which runs PyTorch on CUDA giving me the option to run multiple image generation processes in parallel on my GPU.


The specifications of the PC used was a **RTX 4090**, a **Ryzen 7950x Processor** with **128GB** of **DDR5 Memory**. The setup can be replicated on different setups. 

For creating **512x512px** images, the model needs a base amount of around **4-6 GB** and works with **float16** half-precision floating-points, a format common in other AI and machine learning tasks for its increased efficiency for model training and inference. With the peak of the memory-usage being around **6-8 GB** during image generation. While this is pretty manageable even on mid-range home systems, with higher image resolution the memory use will shoot up considerably so that the pipeline size might only be manageable on DRAM instead of GPU Memory, having to rely on the CPU for inference which slows the generation down considerably. At the end it is a compromise between speed and quality.

### Spanning the Grid in Latent Space

To analyze the [[Latent space]] the authors span a grid in which they sampled 60.000 images. They created 3 random Latents in the space ($z0, z1, z2$) from which they created a grid by calculating the fourth point $z$ through vector addition of $(z0, z1)$ and $(z1, z2$). Then they sampled the grid-points by iteration through vector scalars $α$ and $ß$ to get different position values for $z$.


I proceeded by generating 3 random Latents. A Latent is a compressed numerical representation of an Image in the [[Latent space]]. It is **1/8** output image, which means a **64x64** [[Latent]] results in a **512x512px** image. We can decode this [[Latent]] through a [[Diffusion]] Pipeline.

I start by creating 3 random number generators with fixed seeds for repeatability. The 3 [[Latent|Latents]] created from these have the shape `[1, 64, 64, 4]` which represents a batchsize of 1, the resolution, as well as 4 [[Variational Autoencoder (VAE)|VAE]] feature maps containing the [[Gaussian distribution|gaussian]] [[Noise]] which the [[Diffusion]] Pipeline decodes.

This structure gets permuted into `[1, 4, 64, 64]` which is the Latent-Shape a [[Stable Diffusion]] Pipeline expects.

```
# Core Paramters
latent_height, latent_width = 64, 64
latent_channels = 4

# Fixed Seeds
gen0 = torch.Generator(device=device).manual_seed(0)
gen1 = torch.Generator(device=device).manual_seed(123)
gen2 = torch.Generator(device=device).manual_seed(200)

# Random Latents
z0 = randn_tensor((1, latent_height, latent_width, latent_channels)
                  generator=gen0,
                  device=device,
                  dtype=torch.float16)

z1 = randn_tensor((1, latent_height, latent_width, latent_channels), 
                  generator=gen1, 
                  device=device, 
                  dtype=torch.float16)

z2 = randn_tensor((1, latent_height, latent_width, latent_channels), 
                  generator=gen2, 
                  device=device, 
                  dtype=torch.float16)
                  
# Permuting Latent Shape for Stable Diffusion
z0 = z0.permute(0, 3, 1, 2)
z1 = z1.permute(0, 3, 1, 2)
z2 = z2.permute(0, 3, 1, 2)
```

![[seeds.png|400]]


Out of the prepared Latents we can form a triangle. 

![[forming_a_triangle.png|350]]

And we can already decode the Latents in the Pipeline based on the parameters the authors mentioned in the paper, which interestingly enough gives us pictures of cats and mountains similar to the paper, even though we have not specified the same point locations as in the paper.

![[first_latents.png|350]]

Through a linear combination of $(z0, z1) + (z0, z2)$ we can transform the triangle into a parallelogra. And by introducing the [[scalar|scalars]] $α$ and $ß$ for the vectors $(z0, z1)$ and $(z1, z2)$ respectively we gain the ability to parameterize the space to find any point $z$ on the parallelogram through a [[Baryzentrische Kombination|bayrcentric combination]] of its vertices. 

> [!WICHTIG]
> Erwähne hier mal die linearkombination genau!!

![[bayersic_combination.png|500]]

When decoding point $z$ we stumble on the first hurdle. The reason the image looks this way is because the Latent has to be normalized to the right probability range to be decoded into a plausible image.

![[decoding_z.png|500]]

To fully understand what this normalization process does and to which point exactly the Latent gets normalized it is sensible to look into the math behind it. 

To skip the explanation and get to the next step please go to [Chapter].

## Understanding the Gaussian Hypersphere

> [!Perplexity]
> **Nein, das trainierte Manifold ist keine einfache "Wahrscheinlichkeitsrange", sondern eine hochdimensionale, gekrümmte Struktur** im Gauß-Raum, auf der die echten (denoisierbaren) Datenverteilungen liegen.
> 
> - **Gauß-Raum**: `torch.randn()` füllt **alles** gleichmäßig.
>   
> - **Manifold**: Nur Punkte nahe der echten Datenverteilung sind sinnvoll – dein 0.5z1+0.5z20.5z1+0.5z2 liegt **außerhalb**
>   
> - **Datenmanifold** ℳ ⊂ 𝕉  # 99% "leerer" Raum + dünne "Schicht" mit Bildinfo
>   
>   

```
	    Gauß-Raum (16kD)
       ╭───────────────╮
       │     leeres    │
       │    Volumen    │  ← 99.999% deines z
  z₁───┤  ℳ (Manifold)├───z₂
       │               │
       ╰───────────────╯
```


The authors mention this problem in their paper.

That diffusion models like [[Stable Diffusion]] like to work in a certain probability range in a Gaussian probability space.

To understand what this means lets look at a two dimensional Gaussian distribution. This is the normal distribution or bell curve as we know it. When we mark a certain value on it and ask the question where on the distribution this exact value also occurs, we get a slice through it. If you think about how many dimensions this value has, the answer would be only one dimension.

![[2D_Gauss_4x.png|300]]

If we go one dimension up, we have a three dimensional Gaussian distribution. If we again pick a value and ask for every point where the value is identical. We get a two dimensional value range.

![[3D_gauss_4x.png|300]]
Increasing the dimension to 4D we start to lose the ability to depict the distribution. But the value range has now become a sphere.

![[4D_gauss_4x.png|300]]

Now what if we go even higher? Because [[Stable Diffusion]] doesn't just have 4 dimensions. If we account for the number of feature maps `4` and the resolution of the latent in height and width `64x64` we get a dimension of: $4*64*64 = 16384$ 

If we extend the pattern of the previous examples, the value range was always one dimension lower than its Gaussian distribution. So the value range has a dimension of $16384-1$.

![[dimension-1_title.png]]

If we want to depict this value range, we can again depict it as a sphere, for the simple reason that, if the value is a vector from a center point 0 and it has the same distance in every direction, it can only be a sphere, even in a space with over 16 thousand dimensions where the value range would be an incomprehensible manifold. The term for this structure is a Gaussian Hypersphere or n-sphere.

![[hypersphere_stable_diffusion.png|300]]

If we now normalize our Latent vector so that it falls on this value range, we can decode it to an image.


![[z_norm.png|400]]


![[grid_functioning.png|400]]


Here arises another question. If it needed normalization for this Latent to the Gaussian Hypersphere for it to reveal a plausible image, why were the first 3 Latents on it in the first place?

```
gen0 = torch.Generator(device=device).manual_seed(0)

z0 = randn_tensor((1, latent_height, latent_width, latent_channels)
                  generator=gen0,
                  device=device,
                  dtype=torch.float16)
```

This is the Latent that is constructed. A `randn_tensor()` usually has an even chance to be created at every point in a Gaussian probability space $𝒩(0,I)$.

According to [[Gaussian Annulus Theorem]] a high-dimensional space like the [[Latent space]] of [[Stable Diffusion]], a spherical Gaussian distribution does not behave like a low-dimensional space like a 2D bell-curve.

While the probability mass in low-dimensional Gaussian distributions is located in the center. On higher dimensions, almost all of the probability mass is concentrated in a thin shell (or annulus) around the center at a radius approximately $\sqrt d$ from the center.

As the dimension grows the shell becomes thinner compared to the radius. This "soap bubble" phenomenon describes that even if a Gaussian distribution is located at the center, the vast majority of points sampled will lie on the sphere of radius $\sqrt d$.

That means a `randn_tensor()` created has a very high chance to be located on this annulus. If we take the root of the dimension of the [[Stable Diffusion]] [[Latent space]] we get the radius of that annulus: $$\sqrt 16384 = 128$$
Using the `tensor.norm()` function, we can calculate the magnitude for our random Latents $z0,z1,z2$ in the [[Latent space]], which gives us values very close to the theoretical prediction of the Gaussian annulus.

$z1 = 128.1250$
$z2 = 127.1250$
$z3 = 127.3125$

While if we look at the magnitude of point $z$ calculated through linear combination we get a value of $z = 219.8750$, very much outside the annulus where [[Stable Diffusion]] can generate plausible images.

If we now normalize this value, which means projecting the Latent vector back to the Gaussian annulus, or Gaussian hypersphere as discussed earlier, we [[Stable Diffusion]] can decode it into a plausible image.

## Sampling the Grid (skipping gaussian hypersphere)

> [!NOTE]
> Hier musst du noch reinschreiben wie man punkt z normalisiert falls leute das hypersphere ding skippen

Now with the point $z$ normalized a grid is created which can now be analyzed by scattering points on it. While the authors of the hessian geometry paper used 60.000 to analyze the space, my maximum at home was 2.500 which gave me interesting results.

![[grid_functioning.png|400]]

Through the [[Baryzentrische Kombination|bayrcentric combination]] any point $z$ on the grid can be calculated and decoded into an image. By moving gridwise in a nested loop over the plane any resolution of point grid can be sampled.

```
# Grid parameters
sampling_steps = 5
alphas = np.linspace(0, 1, sampling_steps) # alpha interval (rows/x)
betas = np.linspace(0, 1, sampling_steps) # beta interval (colums/y)

# Capturing latents and their metadata
latent_list = []
metadata_list = []

# Sampling latents
for idx_a, a in enumerate(alphas):
    for idx_b, b in enumerate(betas):
        
        # Sample latent
        z_sample = z0 + a * (z1 - z0) + b * (z2 - z0)
        
        # Normalize latent
        z_normalized = z_sample * (target_norm / z_sample.norm())
        
        # Create latent metadata
        latent_list.append(z_normalized)
        metadata_list.append({
            "filename": f"a{idx_a:03d}_b{idx_b:03d}.png",
            "row": idx_a, 
            "col": idx_b,
            "alpha": float(a), 
            "beta": float(b)
        })
```

After creating a grid of `5x5` latents, we can decode them in the pipeline with the settings of the hessian geometry paper.

```
# Pipeline configuration
pos_prompt = "High quality picture, 4k, detailed"
neg_prompt = "blurry, ugly, stock photo"
inference_steps = 50
guidance = 5.0
pipe_seed = 42 # deterministic seed for reproducibility 
pipe_generator = torch.manual_seed(pipe_seed)

for batch_idx, (latent_batch_linked, metadata_batch) in enumerate(batch_list):
    image_batch = pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        latents=latent_batch_linked,
        num_images_per_prompt=1,
        num_inference_steps=inference_steps,
        guidance_scale=guidance,
        generator=pipe_seed
    )
```

> [!NOTE]
> Das musst du nochmal etwas aufräumen! am besten ein neues Jupyter Lab erzeugen!

![[Fractal_analysis.png|400]]

The result is for this example a 5x5 grid of images that can be sampled at a smaller level for deeper analysis. By changing the start and end of the intervals for $α$ and $ß$ we can sample a smaller space inside the grid. 

Example adapted for sampling a 5x5 grid in the center:

```
sampling_steps = 5
alphas = np.linspace(0.4, 0.6, sampling_steps) # alpha interval (rows/x)
betas = np.linspace(0.4, 0.6, sampling_steps) # beta interval (colums/y)
```


## Batching and Optimization

### Batching

To more efficiently analyse the space I started looking into batching to utilize the GPU to decode multiple latents in parallel. Internally the pipeline then denoises all latents in the current batch in parallel in one U-Net run per step. For a GPU with 24 GB VRAM a batchsize of 8-12 is a good sweetspot. This method decreased the reverse diffusion process per image from 2 seconds per image to around 0.3 seconds for the current setup.

![[batching_1.png|300]]


By concatenating the latents along their first dimension reserved for batches `[1, 4, 64, 64]` for batches of 8 latents, the structure `[8, 4, 64, 64]` is formed. These batches get appended with the metadata of their latents to a list. 

```
batch_size = 8

batch_list = [] # Tuple of latents + metadata


for batch_idx in range(0, len(latent_list), batch_size):
    
    latent_batch = latent_list[batch_idx : batch_idx + batch_size]
    metadata_batch = metadata_list[batch_idx : batch_idx + batch_size]
    
    latent_batch_linked = torch.cat(latent_batch, dim=0) # concatenate latents 

    batch_list.append((latent_batch_linked, metadata_batch)) # Auf Liste schreiben

```

For this `batch_list` a tuple is created with the following structure:

```
batch_list = [
>     (tensor_batch_1,  metadata_batch_1),  # Batch 1
>     (tensor_batch_2,  metadata_batch_2),  # Batch 2  
>     (tensor_batch_3,  metadata_batch_3),  # Batch 3
>     ...
> ]
```

Index 0 contains the batch tensor with the structure `[8, 4, 64, 64]` and index 1 is a list of 8 metadata-strings. The Pipeline then takes both elements per run. 

To avoid token mismatch between the latents and the text-prompts for [[U-Net]] cross-attention, the token amount of the text-embeddings need to match the tokens of the batch. For a single latent has $1*64 * 64*= 4096$ tokens matching the text-embeddings $4096$ tokens. A batch of 8 latents has $8*64*64=36864$, the text-embedding tokens have to match that. For a dynamic pipeline we increase the `num_images_per_prompt` to the `batch_size` of the latent tensor.

```
# Pipeline configuration
pos_prompt = "High quality picture, 4k, detailed"
neg_prompt = "blurry, ugly, stock photo"
inference_steps = 50
guidance = 5.0
pipe_seed = 42 # deterministic seed for reproducibility 
pipe_generator = torch.manual_seed(pipe_seed)

# Pipeline
for batch_idx, (latent_batch_linked, metadata_batch) in enumerate(batch_list):
    image_batch = pipe(
        prompt=pos_prompt,
        negative_prompt=neg_prompt,
        latents=latent_batch_linked,
        num_images_per_prompt=latent_batch_linked.shape[0], # accounts for batchsize
        num_inference_steps=inference_steps,
        guidance_scale=guidance,
        generator=pipe_seed
    )
```

The resulting images get exported to drive and a metadata.json file keeps track of global data like grid resolution, prompts used and positional data for latents in the grid.

![[folder and metadata.png|400]]

### Visualization

This data can be visualized in a Matplotlib plot, with additional labels keeping track of positional data measured both in the grid interval in x and y direction and the values of $α$ and $ß$ for that point.

![[plot_system.jpg|400]]

> [!NOTE]
> Das könntest du nochmal samplen dass du auch das interval von 0 bis 1 hast


## Exploration and Lipschitz Constant

By increasing the grid resolution, more details get revealed. At a resolution of 10x10 images the first anomaly is discovered. In my interpretation the model has failed here to generate a plausible image. It seems like an image that shares features of both mountains and cats without being either. This could be the fractal rift indicative of a phase transition the paper was describing.

![[10x10 grid.png|400]]

![[erste anomalie 3.png|400]]

Increasing the resolution to 50x50 images reveals the structure even more. We see a clear border between two concepts. Not only that but a clear rift is arising at the top of the grid where the anomaly was spotted which revealed images where the model clearly struggled to form a coherent concept.

The authors used different methods to analyse the grid geometry. 

To detect the distinct phases and their transitions they trained a neural network on the log-partition function to reconstruct the fisher information metric of the space, which had discontinuities along the borders between concepts. Through this pathway they were able to calculate geodesic paths through these phases to achieve a smooth interpolation.

To quantify the instability of the model at these transitions they also used the [[Lipschitz-Konstante|Lipschitz constant]], which is a value that tracks the rate of change of a function. At the phase boundaries this constant diverges. Which means even small changes in the latent vector (to a scale of $10^{-8}$ which is at the bit-level of half-precision floating points for the model) are enough to result in extreme fluctuations of the image output between concepts.

![[Lipschitz_Visualisierung.gif|300]]

![[lipschitz_stable.png|200]]


To measure this constant across a grid, its image output is fed into a CLIP model to extract their image features and give every image its CLIP-Embedding, a sort of semantic fingerprint describing the content of the image.

![[clip_fingerprint_2.png|200]]

![[clip_fingerprint.png|400]]

These fingerprints are formed into a feature map, that resembles the coordinates of the original grid. This feature map is then analyzed for local gradient magnitudes across every horizontal and vertical axes, which is used to calculate the [[Lipschitz-Konstante|Lipschitz constant]] of the local changes between images.

![[Lipschitz_at_Fractal.png|400]]

In this resulting field every image cell has a Lipschitz amplitude towards their neighbors. If we take the mean value of every neighbor we have a single value for each cell compared to its neighbors. This value can be plotted over the original image grid.

![[amplitude_grid_plot.png|400]]

![[lipschitz_grid_50x50.png|400]]

The plot reveals in clear detail the points of divergence between the phases. A border ist formed between the mountain manifold and the cat manifold. The highest value being at the rift where the model breaks down and generates unplausible images.

These resulting structures are quite stunning to look at. They were reminding me of mappings of cosmic voids. Huge, nearly galaxy-free bubbles in the universe which are surrounded by large galaxy clusters which result through gravitational instabilities that cause matter to clump into dense structures.

![[cosmic voids.jpg]]

**Image:** Light distribution of galaxies generated in the Millennium Simulation. [Max Planck Institute for Astrophysics]

## Image to Latent

From that point I was interested in using this structure to span different grids to analyse their intersections. A thing I wanted to try is converting existing images into latents to form the grid verteces. 

The paper [[Probability Density Geodesics in Image Diffusion Latent Space]] discusses similar conundrums of the inherent geometry of the latent space. They developed a method for transferring existing images into latent space and interpolating between them.

I was very amused by their examples of interpolating between the actor Dwayne "the Rock" Johnson and a literal Rock.

![[the rock to rock.png]]

During this research I talked with a friend about our work who teaches art in middle school.

The introduced me to a project she did with her class called "Symbiosis." Every kid got to choose two patches of different textures on paper and had to glue them on a sheet of paper. Then every kid had to draw the space in-between those textures, basically a form of interpolation.

[[Kunstunterricht_Ideen_Symbiose.pdf]]

![[symbiose (8).jpeg|300]]

What sparked my interest was the idea of taking these patches and feeding them into the latent space to compare my interpolations with the ones of the class.

There I stumbled onto a wall that I could not overcome yet. It seemed that either the encoding or decoding process had some trouble. I tried different methods, even just importing the original cat and mountain images of a grid into the latent space, it seemed that for this time after a lot of troubleshooting I had to shift to a different approach. I will surely return to this once I have the necessary knowledge.

![[grid_plot_20260210_023309.png|300]]


## Exploring Latent Spcae

I decided to try a different route and use the existing parallelogram formed with the random latents ($z0, z1, z2, z$) and shift the very neutral prompts of the hessian geometry paper away to something more tangible.

To see if I can find model failures or interesting images in these spaces.

I tried prompting for textures and ornaments.

![[grid_10x10_ornaments.png|400]]


What sparked my interest the most was extending the metaphor of discovering the latent space being similar to discovering the universe like a satellite.

One of my favorite videos on the internet is an uncut stream of every of the 341,805 images the Cassini Satellite took on its flight to Saturn between 2004 and 2015. The footage has a mesmerizing quality and eeriness to it, really depicting space as a space almost lovecraftian.

https://www.youtube.com/watch?v=4c8eSr7x7AA

With these depicted prompts I searched through the grid to see what I can discover. Trying to get to the fringes of the model. I started extending the [[scalar|scalars]] $a$ and $b$ close to 10 times, sometimes 1000 times their size, which resulted in that the latents could not even be reprojected to the Gaussian annulus to produce a coherent image. A place where - if I might stay in the metaphore - we have escaped the observable universe.

```
**cassini_pos** = "NASA Cassini spacecraft image of Saturn in greyscale with intricate rings casting sharp shadows, distant moons like Enceladus and Titan visible, realistic colors, scientific photography, high resolution, from orbit, dramatic lighting from sunlight, vast space background, artifacts"

**cassini_neg** = "centered subject, stock photo, person, ornament, fantasy, colorful, flashy"

```

![[out_of_latent_space.png|400]]

Some of the resulting images had an uncanny beauty to them in my opinion. 

















---



> [!Perplexity]
> `torch.norm()` berechnet die **L2-Norm** (euklidische Norm) eines Tensors und misst dessen "Länge" oder Magnitude im Vektorraum.
> 
> Für einen Latent-Vektor `z` (z. B. Shape `(1, 4, 64, 64)`) gibt `torch.norm(z)` den **skalaren Radius** zurück:


- **Gauß-Raum**: `torch.randn()` füllt **alles** gleichmäßig.


> [!Perplexity]
> Ja, `z0` entsteht zunächst im **allgemeinen Gauß-Raum** (dem Zentrum der standardisierten Gauß-Verteilung, also reines Rauschen mit Radius nahe 0). Erst der Reverse-Diffusionsprozess (Denoisierung) zieht es schrittweise zum **Gaussian Annulus** hin, wo valide Latent-Vektoren liegen, die zu hochwertigen Bildern dekodiert werden.





$R4×64×64$

![[Pasted image 20260217160827.png]]



```
# Core Paramters
latent_height, latent_width = 64, 64
latent_channels = 4

# Fixed Seeds
gen0 = torch.Generator(device=device).manual_seed(0)
gen1 = torch.Generator(device=device).manual_seed(123)
gen2 = torch.Generator(device=device).manual_seed(200)

# Random Latents
z0 = randn_tensor((1, latent_height, latent_width, latent_channels)
                  generator=gen0,
                  device=device,
                  dtype=torch.float16)

z1 = randn_tensor((1, latent_height, latent_width, latent_channels), 
                  generator=gen1, 
                  device=device, 
                  dtype=torch.float16)

z2 = randn_tensor((1, latent_height, latent_width, latent_channels), 
                  generator=gen2, 
                  device=device, 
                  dtype=torch.float16)
                  
# Permuting Latent Shape for Stable Diffusion
z0 = z0.permute(0, 3, 1, 2)
z1 = z1.permute(0, 3, 1, 2)
z2 = z2.permute(0, 3, 1, 2)
```





















---

## Ideen

Baue Grids mit [[Nuclear Mysticism]] ein.