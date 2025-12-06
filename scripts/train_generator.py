"""
Trains a flow matching model on astroPT embeddings.

python3 -m scripts.train_generator \
    --model_path <astropt_model_path> \
    --nb_points 14000 --epochs 5000 --sigma 1.0
"""



if __name__ == "__main__":
    import numpy as np
    import argparse, wandb
    import matplotlib.pyplot as plt

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from dataclasses import asdict
    from scripts.model_utils import VectorField, VectorFieldConfig, LinearRegression, load_astropt_model
    from scripts.plot_utils import plot_cross_section_histogram
    from scripts.embedings_utils import merge_datasets, compute_embeddings

    parser = argparse.ArgumentParser(
                    prog='GenAstroPT',
                    description='Trains a flow matching model on astroPT embeddings')
    parser.add_argument('--nb_points', type=int, default=1000, help='Number of points to use for embeddings')
    parser.add_argument('--labels', nargs='+', default=["mass", "label"], help='Labels to use for the conditions of the flow matching model')
    parser.add_argument('--model_path', type=str, default="model/ckpt.pt", help='Path to the astropt checkpoint')
    parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
    parser.add_argument('--sigma', type=float, default=0.01, help='Sigma value for the flow matching model')

    parser.add_argument('--use_wandb', action='store_true', help='Whether to use wandb for logging')
    args = parser.parse_args()
    
    has_metals = torch.backends.mps.is_available()  
    device = torch.device('mps' if has_metals else 
                          'cuda' if torch.cuda.is_available() else 
                          'cpu')
    
    print(f"Generating embeddings on device: {device}")

    model_name = args.model_path.split('/')[-1].replace('.pt','')
    sigma_name = f"sigma_{args.sigma:.3f}".replace('.','_')

    # ==============================================
    # Setup Wandb
    # ==============================================

    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            entity="matttsss-epfl",
            project="Embedding Generation FM",
            name=f"Sigma: {args.sigma:.3f}",
            config={
                "epochs": args.epochs,
                "nb_points": args.nb_points,
                "conditions": args.labels,
                "sigma": args.sigma,
                "astropt_model": model_name
            }
        )
    
    # ==============================================
    # Load astroPT model and compute embeddings
    # ==============================================

    model = load_astropt_model(args.model_path, device=device, strict=True)
    dataset = merge_datasets([
        "data/DarkData/BAHAMAS/bahamas_0.1.pkl", 
        "data/DarkData/BAHAMAS/bahamas_0.3.pkl", 
        "data/DarkData/BAHAMAS/bahamas_1.pkl",
        "data/DarkData/BAHAMAS/bahamas_cdm.pkl"]) \
            .select_columns(["images", "images_positions", *args.labels]) \
            .shuffle(seed=42) \
            .take(args.nb_points)    

    dl = DataLoader(
        dataset,
        batch_size = 64 if has_metals else 256,
        num_workers = 0 if has_metals else 4,
        prefetch_factor = None if has_metals else 3
    )

    embeddings, cond_dict = compute_embeddings(model, dl, device, args.labels)
    cond = torch.stack([cond_dict[k] for k in args.labels], dim=-1)
    nb_embeddings = embeddings.size(0)
    embeddings_dim = embeddings.size(-1)

    # Split into train and val
    nb_train = int(0.8 * nb_embeddings)
    print(f"Training flow matching model on {nb_train} embeddings of dimension {embeddings_dim}.")
    print(f"Validation on {nb_embeddings - nb_train} embeddings.")

    train_embeddings = embeddings[:nb_train]
    val_embeddings = embeddings[nb_train:]
    train_cond = cond[:nb_train]
    val_cond = cond[nb_train:]

    del embeddings, cond, cond_dict

    # ==============================================
    # Train Flow Matching model
    # ==============================================

    train_embed_dl = DataLoader(TensorDataset(train_embeddings, train_cond), 
                                batch_size=nb_train)
    val_embed_dl = DataLoader(TensorDataset(val_embeddings, val_cond), 
                              batch_size=nb_embeddings - nb_train)
    
    fm_config = VectorFieldConfig(
        sigma=args.sigma,
        dim=embeddings_dim,
        hidden=512,
        conditions=args.labels
    )
    fm_model = VectorField(fm_config).to(device)
    opt = torch.optim.AdamW(fm_model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    epoch_val_losses = []
    epoch_train_losses = []
    for epoch in range(args.epochs):
        train_loss = 0.0
        for x1, cond in train_embed_dl:
            x1 = x1.to(device).view(x1.size(0), -1)
            cond = cond.to(device).view(cond.size(0), -1)

            x0 = torch.randn_like(x1)
            loss = fm_model.compute_loss(x0, x1, cond)

            opt.zero_grad()
            loss.backward()
            opt.step()
                
            train_loss += loss.item()

        train_loss /= len(train_embed_dl)

        val_loss = 0.0
        for x1, cond in val_embed_dl:
            x1 = x1.to(device).view(x1.size(0), -1)
            cond = cond.to(device).view(cond.size(0), -1)

            x0 = torch.randn_like(x1)
            loss = fm_model.compute_loss(x0, x1, cond)

            val_loss += loss.item()

        val_loss /= len(val_embed_dl)
        
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)

        if epoch % 10 == 0:
            if wandb_run is not None:
                wandb_run.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })
            else:
                print(f"Epoch {epoch}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                        "state_dict": fm_model.state_dict(),
                        "config": asdict(fm_config),
                        "conditions": fm_config.conditions
                    }, 
                    f"model/flow_matching/{model_name}.pt")
    
    # ==============================================
    # Make predictions for plots
    # ==============================================

    lin_reg = LinearRegression(device).fit(train_embeddings, train_cond)
    lin_preds = lin_reg.predict(val_embeddings).cpu().numpy()

    vf_embeddings = fm_model.sample_flow(val_cond, steps=int(1e4))
    vf_preds = lin_reg.predict(vf_embeddings).cpu().numpy()

    lin_preds = {label_name: lin_preds[:, i] for i, label_name in enumerate(fm_model.config.conditions)}
    vf_preds = {cond_name: vf_preds[:, i] for i, cond_name in enumerate(fm_model.config.conditions)}
    val_cond = {label_name: val_cond[:, i].cpu().numpy() for i, label_name in enumerate(fm_model.config.conditions)}


    # ==============================================
    # Plot predictions (except for cross sections)
    # ==============================================

    for cond_name in filter(lambda x: "label" not in x, args.labels):
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))

        rel_diff = np.abs(lin_preds[cond_name] - val_cond[cond_name]) / np.maximum(np.abs(val_cond[cond_name]), 1e-6)

        lin_ax.scatter(val_cond[cond_name], lin_preds[cond_name], alpha=0.1)
        lin_ax.set_title(f"Linear Regression Predictions for {cond_name}")
        lin_ax.set_xlabel(f"Ground truth {cond_name}")
        lin_ax.set_ylabel(f"Predicted {cond_name}")

        fm_ax.scatter(val_cond[cond_name], vf_preds[cond_name], alpha=0.1)
        fm_ax.set_title(f"Flow Matching Predictions for {cond_name}")
        fm_ax.set_xlabel(f"Ground truth {cond_name}")
        fm_ax.set_ylabel(f"Predicted {cond_name}")

        fig.tight_layout()

        if wandb_run is not None:
            wandb_run.log({f"{cond_name}_predictions": wandb.Image(fig)})
        else:
            fig.savefig(f"figures/{model_name}_{sigma_name}_{cond_name}_predictions.png", dpi=300)
        
        plt.close(fig)
    
    # ==============================================
    # Plot cross-section for label if available
    # ==============================================

    if "label" in args.labels:
        fig, (lin_ax, fm_ax) = plt.subplots(1, 2, figsize=(12, 6))

        plot_cross_section_histogram(lin_ax,
            val_cond["label"], lin_preds["label"], 
            pred_method_name="Linear Regression")
        
        plot_cross_section_histogram(fm_ax,
            val_cond["label"], vf_preds["label"], 
            pred_method_name="Flow Matching + Linear Regression")
        
        if wandb_run is not None:
            wandb_run.log({f"label_predictions": wandb.Image(fig)})
        else:
            fig.savefig(f"figures/{model_name}_{sigma_name}_label_predictions.png", dpi=300)
        
        plt.close(fig)