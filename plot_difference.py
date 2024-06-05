import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
from torchvision import transforms as tfms
from PIL import Image

# Function to visualize the features and their differences with global normalization
def visualize_features_and_differences(feature1, feature2, save_name, save_dir='./plots'):
    feature1 = feature1.squeeze(0)  # Remove the batch dimension
    feature2 = feature2.squeeze(0)  # Remove the batch dimension
    differences = torch.abs((feature1 - feature2)**2)  # Compute absolute differences

    # Normalize differences globally
    max_diff = differences.max().item()
    min_diff = differences.min().item()
    # norm_differences = differences / max_diff

    fig, axes = plt.subplots(4, 3, figsize=(18, 18))

    for i in range(4):
        # Plot feature1
        ax1 = axes[i, 0]
        ax1.imshow(feature1[i], cmap='viridis')
        ax1.set_title(f'Latent of Luminance - Channel {i+1}')
        ax1.axis('off')

        # Plot feature2
        ax2 = axes[i, 1]
        ax2.imshow(feature2[i], cmap='viridis')
        ax2.set_title(f'Latent of RGB - Channel {i+1}')
        ax2.axis('off')

        # Plot normalized differences
        ax3 = axes[i, 2]
        cax = ax3.imshow(differences[i], cmap='hot', vmin=min_diff, vmax=max_diff)
        ax3.set_title(f'Difference - Channel {i+1}')
        ax3.axis('off')

        # Add color bar to show the scale of differences
        fig.colorbar(cax, ax=ax3)

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{save_dir}/'+save_name)
    
def get_latent(pipe, img_path):
    vae_magic = 0.18215
    img = Image.open(img_path).convert('RGB').resize((512, 512))
    img = tfms.ToTensor()(img).unsqueeze(0).to(pipe.device, pipe.dtype)
    with torch.no_grad():
            latents = pipe.vae.encode(img.to(device=pipe.device)*2 -1)
    latents = latents['latent_dist'].mean * vae_magic
    return latents
        
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    use_safetensors=True,
).to('cuda')

img_paths = ['sample/banana.jpg', 'sample/bear.jpg', 'sample/giraffe.jpg', 'sample/man.jpg']

for img_path in img_paths:
    img_path_l = img_path.replace('.jpg', '_l.jpg')
    save_name = img_path.split('/')[-1].split('.')[0]
    save_name = 'latent_l2_diff_' + save_name + '.png'

    latent_l = get_latent(pipe, img_path_l)
    latent = get_latent(pipe, img_path)

    visualize_features_and_differences(latent_l.cpu(), latent.cpu(), save_name)