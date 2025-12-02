import argparse
from generative_model.DDPM import DDPM
from generative_model.DiT import DiT
import pickle 
import torch
import wandb

def training_script(dataset_path, output_dir):

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="pijukai-epfl",
    # Set the wandb project where this run will be logged.
    project="astropt_diffusion",
    name="Diffusion_Training")
    
    data = torch.load(dataset_path)
    diffusion_model = DDPM()

    dataloader_train = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

    epochs = 50

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=3e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_validation_loss = float('inf')

    for epoch in range(epochs):

        diffusion_model.train()

        optimizer.zero_grad()

        loss = 0

        for batch in dataloader_train:
            x_0 = batch['data']
            conditions = batch['conditions']

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

            loss_eval = 0
            diffusion_model.eval()

            for batch in dataloader_val:
                x_0 = batch['data']
                conditions = batch['conditions']

                t = torch.randint(0, diffusion_model.timesteps, (x_0.size(0),), dtype=torch.long)

                noise = torch.randn_like(x_0)

                x_t = diffusion_model.q_sample(x_0, t, noise)
                noise_pred = diffusion_model.eps_model(x_t, t, conditions)

                loss_eval +=torch.nn.functional.mse_loss(noise_pred, noise)
            
            print(f"Epoch {epoch}, Val Loss: {loss_eval.item()}")
            run.log({"epoch": epoch, "loss_val": loss_eval.item()})

            if loss_eval < best_validation_loss:
                best_validation_loss = loss_eval
                print("Saving best model...")
                torch.save(diffusion_model.state_dict(), output_dir + '/best_diffusion_model.pt')
                print("Best model saved.")



if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset")
    args.add_argument("--output_dir", type=str, required=True, help="Output directory to save the trained model")
    args = args.parse_args()
    
    training_script(args.dataset_path, args.output_dir)



