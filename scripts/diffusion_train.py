import argparse
from generative_model.DDPM import DDPM
from model_utils import load_model
from embedings_utils import merge_datasets,compute_embeddings, labels_name
import torch
from torch.utils.data import TensorDataset, random_split
import wandb

def training_script(output_dir,weights_path):

    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="pijukai-epfl",
    # Set the wandb project where this run will be logged.
    project="astropt_diffusion",
    name="Diffusion_Training")

    model = load_model(weights_path, device=device, strict=False)


    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *labels_name]) \
            .shuffle(seed=42) \
            .take(args.nb_points)    

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size = 128,
        num_workers = 4,
        prefetch_factor = 3
    )

    embeddings, labels = compute_embeddings(model, dl, device, ["mass","label"])
    embeddings = embeddings.cpu()
    labels = {k: v.cpu() for k, v in labels.items()}

    dataset_embeddings = torch.utils.data.TensorDataset(embeddings, labels["mass"], labels["label"])

    
    train_size = int(0.8 * len(dataset_embeddings))
    val_size = len(dataset_embeddings) - train_size
    train_dataset, val_dataset = random_split(dataset_embeddings, [train_size, val_size])

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4)
    
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)

    
    diffusion_model = DDPM()

    epochs = 50

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=3e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_validation_loss = float('inf')

    for epoch in range(epochs):

        diffusion_model.train()

        optimizer.zero_grad()

        loss = 0

        for batch in dataloader_train:
            x_0 = batch[0]
            conditions = batch[1:]

            t = torch.randint(0, diffusion_model.timesteps, (x_0.size(0),), dtype=torch.long)

            noise = torch.randn_like(x_0)

            x_t = diffusion_model.q_sample(x_0, t, noise)
            noise_pred = diffusion_model.eps_model(x_t, t, conditions)

            loss += torch.nn.functional.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        run.log({"epoch": epoch, "loss_train": loss.item()})

        if epoch % 2 == 0: 

            loss_val = 0
            diffusion_model.eval()

            for batch in dataloader_val:
                x_0 = batch['data']
                conditions = batch['conditions']

                t = torch.randint(0, diffusion_model.timesteps, (x_0.size(0),), dtype=torch.long)

                noise = torch.randn_like(x_0)

                x_t = diffusion_model.q_sample(x_0, t, noise)
                noise_pred = diffusion_model.eps_model(x_t, t, conditions)

                loss_val +=torch.nn.functional.mse_loss(noise_pred, noise)
            
            print(f"Epoch {epoch}, Val Loss: {loss_val.item()}")
            run.log({"epoch": epoch, "loss_val": loss_val.item()})

            if loss_val < best_validation_loss:
                best_validation_loss = loss_val
                print("Saving best model...")
                torch.save(diffusion_model.state_dict(), output_dir + '/best_diffusion_model.pt')
                print("Best model saved.")



if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--output_dir", type=str, required=True, help="Output directory to save the trained model")
    args = args.parse_args()
    
    training_script(args.output_dir)


