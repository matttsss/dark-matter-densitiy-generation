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
    steps: int = 300,
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
    """image → 768-dim embedding"""
    
    # Handle tensor conversion
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    else:
        image = image.detach().clone()

    print(f"Image shape: {image.shape}")

    # Check if data is already embedded (256, 768) vs raw image
    if image.dim() == 2 and image.shape[-1] == 768:
        # Already tokenized! Just pool over sequence dimension
        embeddings = image.mean(dim=0, keepdim=True).to(device)  # (1, 768)
        print(f"Data already embedded, pooled to: {embeddings.shape}")
        return embeddings

    # Otherwise, run through AstroPT...
    if positions is None:
        raise ValueError("AstroPT requires positions alongside image")

    if not isinstance(positions, torch.Tensor):
        positions = torch.as_tensor(positions)
    else:
        positions = positions.detach().clone()

    model = load_astropt_model(checkpoint_path, device)
    model.eval()

    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)

    with torch.no_grad():
        batch = {
            "images": image.to(device),
            "images_positions": positions.to(device),
        }
        outputs = model.generate_embeddings(batch)

        if isinstance(outputs, dict):
            embeddings = outputs.get("images", outputs.get("embeddings"))
        else:
            embeddings = outputs

        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)

    return embeddings


def load_ddpm(checkpoint_path: str, device: torch.device):
    """Load pretrained DDPM model."""
    from generative_model.DDPM import DDPM
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        raise ValueError(f"Invalid DDPM checkpoint path: {checkpoint_path}")
    
    model = DDPM().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def infer_ddpm(
    ddpm_model,
    conditioning_embeddings: torch.Tensor,
):
    """
    Generate images from conditioning embeddings using DDPM.
    
    Args:
        ddpm_model: Pretrained DDPM model
        conditioning_embeddings: (B, 768) - conditioning vectors   
    Returns:
        (B, 1, 100, 100) - generated images
    """
    device = next(ddpm_model.parameters()).device
    conditioning_embeddings = conditioning_embeddings.to(device)
    
    ddpm_model.eval()
    with torch.no_grad():
        generated_images = ddpm_model.sample(conditioning_embeddings)

    return generated_images

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fm_path", type=str, default="model_weights/fm.pt", help="Path to flow model checkpoint")
    parser.add_argument("--astropt_path", type=str, default="model_weights/finetuned_contrastive_ckpt.pt", help="Path to AstroPT checkpoint")
    parser.add_argument("--ddpm_path", type=str, default="model_weights/best_diffusion_model.pt",help="Path to DDPM checkpoint")
    parser.add_argument("--mass", type=float, help="Mass value for conditioning")
    parser.add_argument("--label", type=float, help="Label value for conditioning")
    parser.add_argument("--sample_idx", type=int, help="Index of sample from BAHAMAS dataset")
    parser.add_argument("--steps", type=int, default=3000, help="Flow model integration steps")
    parser.add_argument("--output_path", type=str, default="generated_images.pt", help="Output path")
    parser.add_argument("--save_png", action="store_true", help="Save as PNG image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sample_idx is not None:
        # Load from BAHAMAS dataset
        dataset = merge_datasets([
            "data/BAHAMAS/bahamas_0.1.pkl",
            "data/BAHAMAS/bahamas_0.3.pkl",
            "data/BAHAMAS/bahamas_1.pkl",
            "data/BAHAMAS/bahamas_cdm.pkl",
        ]).select_columns(["images", "images_positions", "mass", "label"])
        
        sample = dataset[args.sample_idx]
        
        # Use as_tensor to avoid copy warning
        image = torch.as_tensor(sample["images"])
        positions = torch.as_tensor(sample["images_positions"])
        
        print(f"Loaded sample {args.sample_idx}")
        print(f"  mass: {sample['mass']}, label: {sample['label']}")
        print(f"  image shape: {image.shape}, positions shape: {positions.shape}")
        
        # Compute embeddings from image
        embeddings = get_embeddings(
            device,
            image=image,
            positions=positions,
            astropt_path=args.astropt_path,
        )
        
    elif args.mass is not None and args.label is not None:
        # Use mass/label conditioning
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
        for i, img in enumerate(generated_images):
            plt.figure()
            plt.imshow(img[0].cpu().numpy(), cmap="viridis")
            plt.colorbar()
            plt.savefig(f"generated_{i}.png")
            plt.close()
        print(f"Saved {len(generated_images)} PNG images")