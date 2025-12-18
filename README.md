

## What is the project ?

This repository implements the content of the project (detailed in `report.pdf`). In the following sections, we will show you how to train and run inference with the models contained inside the report.

Note that while in the code some scripts support MPS, some still have issues with it. Hence, we strongly recommend using cuda or as a last resort the cpu to run our code.

## Project Structure
```
├── .github/                    # GitHub configuration
├── cache/                      # Cached data and checkpoints
├── data/                       # Dataset storage (BAHAMAS simulations)
├── figures/                    # Generated plots and visualizations
├── generative_model/           # Core model implementations
│   ├── DDPM.py                 # Denoising Diffusion Probabilistic Model
│   ├── DiT.py                  # Diffusion Transformer backbone
│   ├── vector_field.py         # Flow matching vector field network
│   └── __init__.py
├── model_weights/              # Saved model weights
├── scripts/                    # Training and evaluation scripts
│   ├── plots/                  # Plotting utilities
│   ├── diffusion_eval.py       # DDPM evaluation metrics
│   ├── diffusion_train.py      # DDPM training loop
│   ├── embeddings_utils.py     # Embedding extraction utilities
│   ├── finetune_astropt.py     # AstroPT finetuning script
│   ├── flow_model_train.py     # Flow matching training
│   └── model_utils.py          # Shared model utilities
├── inference.py                # Main inference pipeline
├── README.md
├── pip_requirements.txt        # requirements file for venv to run our project
├── download_weights.py         # Downloads data and model weights to the correct folders
├── report.pdf
└── .gitignore
```


## Pre-requisites

Starting from root, you can create the virtual environment with the following command :
```bash
python3 -m venv .venv
```

Then, enter your virtual environment :
```bash
source .venv/bin/activate
```

Following that, download the necessary packages to run the code.
```bash
python -m pip install --upgrade pip && pip install -r pip_requirements.txt
```

Lastly, download the weights and datasets through the following script, it will download them from our Google Drive and place them into the correct folders.
```bash
python download_weights.py
```
If, for some reason, the script does not work, you can download them from:
```https://drive.google.com/drive/folders/1y1Wwj65rAHpNVW2cQMdUe-hwqwgwqVPa?usp=sharing```\
You will then need to move the `BAHAMAS` folder into `data` and extract the model weights from `MODEL_WEIGHTS` into `model_weights`.

Finally, if you wish to run the code please follow the next steps.

## How to train the models
Once the weights are downloaded, you can train/finetune the models in the following ways:

### AstroPT Finetune

```bash
python3 -m scripts.finetune_astropt \
      --pretrained_path model_weights/baseline_astropt.pt \
      --output_path model_weights/finetuned_model.pt \
      --batch_size 64 \
      --num_epochs 60 \
      --learning_rate 1e-5 \
      --contrastive_weight 0.1 \
      --label_names mass label BCG_e1 BCG_e2
```

### Flow Matching
```bash
python3 -m scripts.flow_model_train \
      --model_path model_weights/finetuned_astropt.pt \
      --nb_points 14000 \
      --epochs 5000 \
      --sigma 1.0 \
      --save_plots
```

### Diffusion Model
To train the diffusion model from the AstroPT embeddings, you can run:
```bash
python -m scripts.diffusion_train
```

For training on the flow matching embeddings, you need:
```bash
python -m scripts.diffusion_train --weights_path model_weights/flow_model.pt --mode fm
```

## How to run inference
Here is the command to run in order for the script to do inference. Make sure that the virtual environment is still activated. For further personalization of the inference, you can take a look at the inference script in the code where you will see all the available options and flags

You can then run Inference Mode 1 (FM model) by writing in arguments the mass and label (one of 0.01, 0.1, 0.3, 1). Here’s an example command to run from root:

```bash
python ./inference.py --mass 14.9 --label 0.1 --mode fm
```

You can then run Inference Mode 2 (AstroPT model) by writing in arguments the image index of the image you want to reconstruct (can be between 0 or 14000). Here’s an example command to run from root:

```bash
python ./inference.py --sample_idx 42  --mode astropt
```
You will then see the generated image at generated.png (at root)

## How to make the plots
### AstroPT validation plots

To make the comparison plots, you can run the following:
```bash
python3 -m scripts.plots.finetune_comparison 
      --finetuned_model_path model_weights/finetuned_astropt.pt 
      --baseline_model_path model_weights/baseline_astropt.pt 
      --labels label mass 
      --nb_points 14000
```

And to get more fine-grained plots:
```bash
python3 -m scripts.plots.linear_probe \
      --model_path model_weights/finetuned_astropt.pt \
      --labels label mass BCG_e1 BCG_e2 BCG_stellar_conc \
      --output_path figures/ \ 
      --nb_points 14000
```

### Flow Matching plots

In order to get accuracy plots:
```bash
python3 -m scripts.plots.fm_validation \
      --fm_model_path model_weights/flow_model.pt \
      --astropt_model_path model_weights/finetuned_astropt.pt \
      --labels mass label \
      --nb_points 8000 \
      --save_plots
```

To try specific conditions:
```bash
python3 -m scripts.plots.fm_inference \
      --fm_model_path model_weights/flow_model.pt \
      --astropt_model_path model_weights/finetuned_astropt.pt \
      --nb_points 14000 \
      --nb_gen_points 6000 \
      --labels mass label
```

## References and external libraries

You can find the extensive list of the libraries that were used in this project in pip_requirements.txt.

We also leverage the AstroPT model in this project. Hence, you can find the original pre-trained weights of the model at this link : https://huggingface.co/Smith42/astroPT_v2.0/tree/main/astropt/095M .

For the DiT implementation, we were inspired by the following implementation from Meta : https://github.com/facebookresearch/DiT/blob/main/models.py .





