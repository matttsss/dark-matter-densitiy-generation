import torch
from scripts.model_utils import load_fm_model, load_astropt_model
from scripts.embedings_utils import merge_datasets
import os



def get_embeddings(
    device: torch.device,
    # Option 1: conditioning inputs (→ flow model)
    mass: float = None,
    label: float = None,
    # Option 2: image input (→ AstroPT)
    image: torch.Tensor = None,
    positions: torch.Tensor = None,
    # Config
    steps: int = 5000,
    fm_path: str = "model/flow_ckpt.pt",
    astropt_path: str = "model/finetuned_ckpt.pt",
):
    """
    Get 768-dim embeddings compatible with DDPM.
    
    Automatically selects model based on input:
        - image provided       → AstroPT
        - mass/label provided  → Flow model
    """
    has_image = image is not None
    has_conditioning = mass is not None and label is not None

    if has_image and has_conditioning:
        raise ValueError("Provide either image OR (mass, label), not both")

    if not has_image and not has_conditioning:
        raise ValueError("Provide either image OR (mass, label)")

    if has_image:
        return _embed_with_astropt(device, image, positions, astropt_path)
    else:
        return _embed_with_flow(device, mass, label, steps, fm_path)


def _embed_with_flow(device, mass, label, steps, checkpoint_path):
    """(mass, label) → 768-dim embedding"""
    model = load_fm_model(checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        cond = torch.tensor([[mass, float(label)]], dtype=torch.float32, device=device)
        embeddings = model.sample_flow(cond, steps=steps)

    return embeddings


def _embed_with_astropt(device, image, positions, checkpoint_path):
    """Match exactly what diffusion training did"""
    
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    else:
        image = image.detach().clone()
        
    if not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions)
    else:
        positions = positions.detach().clone()

    # Use the SAME loader as diffusion training
    # Import from the same place as diffusion_train.py
    
    model = load_astropt_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()

    # Ensure batch dimension
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)

    with torch.no_grad():
        batch = {
            "images": image.to(device),
            "images_positions": positions.to(device),
        }
        # Use generate_embeddings - same as training
        embeddings = model.generate_embeddings(batch)["images"]
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"Std: {embeddings.std():.4f}")

    return embeddings


def load_ddpm(checkpoint_path: str, device: torch.device):
    """Load pretrained DDPM model."""
    from generative_model.DDPM import DDPM
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise ValueError(f"Invalid DDPM checkpoint path: {checkpoint_path}")
    
    # Match training config
    model = DDPM(
        patch_size=4,       # Changed from default 10
        schedule="cosine",
        timesteps=1000  # Changed from linear
        # depth=12, timesteps=1000 are defaults
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def infer_ddpm(
    ddpm_model,
    conditioning_embeddings: torch.Tensor,
    clamp_output: bool = True,
    use_ddim: bool = True,        # NEW
    ddim_steps: int = 50,         # NEW
    guidance_scale: float = 3.0,  # NEW
):
    device = next(ddpm_model.parameters()).device
    conditioning_embeddings = conditioning_embeddings.to(device)
    
    ddpm_model.eval()
    with torch.no_grad():
        if use_ddim:
            generated_images = ddpm_model.ddim_sample(
                conditioning_embeddings, 
                steps=ddim_steps, 
                eta=0.0,
                guidance_scale=guidance_scale
            )
        else:
            generated_images = ddpm_model.sample(
                conditioning_embeddings,
                guidance_scale=guidance_scale
            )
        
        if clamp_output:
            generated_images = (generated_images - generated_images.min()) / (generated_images.max() - generated_images.min() + 1e-8)

    return generated_images


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm_path", type=str, default="model_weights/fm.pt", help="Path to flow model checkpoint")
    parser.add_argument("--astropt_path", type=str, default="model_weights/finetuned_contrastive_ckpt.pt", help="Path to AstroPT checkpoint")
    parser.add_argument("--ddpm_path", type=str, default="model_weights/final_diffusion_model_1000.pt", help="Path to DDPM checkpoint")
    parser.add_argument("--mass", type=float, help="Mass value for conditioning")
    parser.add_argument("--label", type=float, help="Label value for conditioning")
    parser.add_argument("--sample_idx", type=int, help="Index of sample from BAHAMAS dataset")
    parser.add_argument("--steps", type=int, default=3000, help="Flow model integration steps")
    parser.add_argument("--output_path", type=str, default="generated_images.pt", help="Output path")
    parser.add_argument("--save_png", action="store_true", help="Save as PNG image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    original_image = None  # Will store original if using sample_idx

    if args.sample_idx is not None:
        # Load embeddings dataset
        dataset = merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ]).select_columns(["images", "images_positions", "mass", "label"])
        
        # Load raw images dataset
        dataset_images = merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ], image_only=True)
        
        sample = dataset[args.sample_idx]
        original_image = np.array(dataset_images[args.sample_idx]["image"])
        
        image = torch.as_tensor(sample["images"])
        positions = torch.as_tensor(sample["images_positions"])
        
        print("=" * 60)
        print("DEBUG: Sample Info")
        print("=" * 60)
        print(f"Sample index: {args.sample_idx}")
        print(f"Mass: {sample['mass']}")
        print(f"Label: {sample['label']}")
        
        print("\n" + "=" * 60)
        print("DEBUG: Original Image Stats")
        print("=" * 60)
        print(f"Shape: {original_image.shape}")
        print(f"Dtype: {original_image.dtype}")
        print(f"Range: [{original_image.min():.6f}, {original_image.max():.6f}]")
        print(f"Mean: {original_image.mean():.6f}")
        print(f"Std: {original_image.std():.6f}")
        
        print("\n" + "=" * 60)
        print("DEBUG: Pre-computed Embeddings (from dataset)")
        print("=" * 60)
        print(f"Shape: {image.shape}")
        print(f"Dtype: {image.dtype}")
        print(f"Range: [{image.min():.6f}, {image.max():.6f}]")
        print(f"Mean: {image.mean():.6f}")
        print(f"Std: {image.std():.6f}")
        
        print("\n" + "=" * 60)
        print("DEBUG: Positions")
        print("=" * 60)
        print(f"Shape: {positions.shape}")
        print(f"Range: [{positions.min():.6f}, {positions.max():.6f}]")
        
        embeddings = get_embeddings(
            device,
            image=image,
            positions=positions,
            astropt_path=args.astropt_path,
        )
        
        print("\n" + "=" * 60)
        print("DEBUG: Pooled Embeddings (input to DDPM)")
        print("=" * 60)
        print(f"Shape: {embeddings.shape}")
        print(f"Range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")
        print(f"Mean: {embeddings.mean():.6f}")
        print(f"Std: {embeddings.std():.6f}")
        
        # Compare with training embeddings if possible
        print("\n" + "=" * 60)
        print("DEBUG: Checking against training data format")
        print("=" * 60)
        
        # Load a few samples to check consistency
        train_embeddings_sample = torch.as_tensor(dataset[0]["images"])
        print(f"Train sample 0 embeddings shape: {train_embeddings_sample.shape}")
        print(f"Train sample 0 embeddings range: [{train_embeddings_sample.min():.6f}, {train_embeddings_sample.max():.6f}]")
        
        # Check if pooling matches what training did
        pooled_train = train_embeddings_sample.mean(dim=0)
        print(f"Train sample 0 pooled shape: {pooled_train.shape}")
        print(f"Train sample 0 pooled range: [{pooled_train.min():.6f}, {pooled_train.max():.6f}]")
        
        print("\n" + "=" * 60)
        
    elif args.mass is not None and args.label is not None:
        embeddings = get_embeddings(
            device,
            mass=args.mass,
            label=args.label,
            steps=args.steps,
            fm_path=args.fm_path,
        )
    else:
        raise ValueError("Provide either --sample_idx or both --mass and --label")

    ddpm_model = load_ddpm(args.ddpm_path, device)
    generated_images = infer_ddpm(ddpm_model, embeddings)

    print("Generated images shape:", generated_images.shape)
    torch.save(generated_images, args.output_path)
    print(f"Generated images saved to {args.output_path}")

if args.save_png:
    for i, gen_img in enumerate(generated_images):
        gen_np = gen_img[0].cpu().numpy()
        
        if original_image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Handle different original image formats
            print(f"Original shape: {original_image.shape}")
            
            if original_image.ndim == 3:
                # Could be (H, W, C) or (C, H, W)
                if original_image.shape[-1] in [1, 3]:
                    # (H, W, C) format
                    orig_display = original_image[:, :, 0]
                elif original_image.shape[0] in [1, 3]:
                    # (C, H, W) format
                    orig_display = original_image[0, :, :]
                else:
                    # Unknown, try first slice
                    orig_display = original_image[:, :, 0]
            elif original_image.ndim == 2:
                orig_display = original_image
            elif original_image.ndim == 1:
                # 1D array - try to reshape to square
                side = int(np.sqrt(len(original_image)))
                if side * side == len(original_image):
                    orig_display = original_image.reshape(side, side)
                else:
                    print(f"Warning: Cannot reshape 1D array of length {len(original_image)} to square")
                    orig_display = original_image.reshape(-1, 100)  # fallback
            else:
                orig_display = original_image
                
            print(f"Display shape: {orig_display.shape}")
            
            # Original
            im0 = axes[0].imshow(orig_display, cmap="viridis")
            axes[0].set_title(f"Original (idx={args.sample_idx})\nShape: {orig_display.shape}")
            plt.colorbar(im0, ax=axes[0])
            
            # Generated
            im1 = axes[1].imshow(gen_np, cmap="viridis")
            axes[1].set_title(f"Generated\nShape: {gen_np.shape}")
            plt.colorbar(im1, ax=axes[1])
            
            # Difference
            from skimage.transform import resize
            if orig_display.shape != gen_np.shape:
                orig_resized = resize(orig_display, gen_np.shape, anti_aliasing=True)
            else:
                orig_resized = orig_display
            
            orig_norm = (orig_resized - orig_resized.min()) / (orig_resized.max() - orig_resized.min() + 1e-8)
            gen_norm = (gen_np - gen_np.min()) / (gen_np.max() - gen_np.min() + 1e-8)
            diff = orig_norm - gen_norm
            
            im2 = axes[2].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
            axes[2].set_title("Difference (Original - Generated)")
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(f"comparison_{i}.png", dpi=150)
            plt.close()