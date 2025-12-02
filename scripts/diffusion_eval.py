import argparse
from typing import Tuple
from generative_model.DDPM import DDPM
import torch


args = argparse.ArgumentParser()
args.add_argument("--condition" , type= Tuple[float,float], required=True, help="Conditions for evaluation")
args.add_argument("--output_dir", type=str, default="diffusion_eval.pt", help="Path to save the generated embeddings")
args = args.parse_args()

condition = args.condition
output_dir = args.output_dir

diffusion_model = DDPM()
diffusion_model.load_state_dict(torch.load('best_diffusion_model.pt'))

diffusion_model.eval()

generated_embedding = diffusion_model.sample(torch.tensor([condition]))


print("Saving generated embeddings...")

torch.save(generated_embedding, output_dir + 'generated_embedding.pt')

print(f"Generated embeddings saved to {output_dir + 'generated_embedding.pt'}")





