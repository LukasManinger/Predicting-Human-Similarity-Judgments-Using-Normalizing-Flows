# Predicting-Human-Similarity-Judgments-Using-Normalizing-Flows
Code for reproducing results in my thesis "Predicting Human Similarity Judgments Using Normalizing Flows" (Cognitive Science B.Sc.).

## Requirements

The `requirement.txt` for a complete list of our Python environment.

The main requirements are:

```
numpy==1.23.*
scipy==1.9.*
pandas==1.4.*
scikit-learn==1.1.*
matplotlib==3.5.*
seaborn==0.11.*
tqdm==4.64.*

psiz==0.7.*
tensorflow==2.8.*
tensorflow-gpu==2.8.*
tensorflow-datasets==4.6.*
tensorflow-probability==0.16.*
keras==2.8.*
opencv-python==4.6.*
```

## Usage

### Train a Normalizing Flow

```
python src/glow/train_glow.py
```

Possible command line arguments:

```python
parser.add_argument("--dataset", default="caltech_birds2011", type=str)
parser.add_argument("--train_slice", default="train+test[:80%]", type=str)
parser.add_argument("--test_slice", default="test[80%:]", type=str)
parser.add_argument("--image_size", default=32, type=int)
parser.add_argument("--central_crop_fraction", default=0.8, type=float)
parser.add_argument("--preprocess_mode", default=3, type=int)
parser.add_argument("--n_bits", default=5, type=int)
parser.add_argument("--not_disable_tqdm", action="store_false")
parser.add_argument("--num_glow_blocks", default=3, type=int)
parser.add_argument("--num_steps_per_block", default=32, type=int)
parser.add_argument("--coupling_bijector_fn", default="GlowDefaultNetwork", type=str)
parser.add_argument("--grab_after_block", default=None, type=tuple)
parser.add_argument("--not_use_actnorm", action="store_false")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--learning_rate", default=1e-5, type=float)
parser.add_argument("--optimizer_clipvalue", default=10.0, type=float)
parser.add_argument("--optimizer_global_clipnorm", default=1.0, type=float)
parser.add_argument("--adam_epsilon", default=1e-7, type=float)
parser.add_argument("--cosine_decay_steps", default=10, type=int)
parser.add_argument("--cosine_decay_t_mul", default=2.0, type=float)
parser.add_argument("--cosine_decay_m_mul", default=1.0, type=float)
parser.add_argument("--cosine_decay_alpha", default=0.0, type=float)
parser.add_argument("--checkpoint_freq", default=10, type=int)
parser.add_argument("--name", default="tfp_glow", type=str)
```

### Train a ResNet

```
python src/resnet/train_resnet.py
```

Possible command line arguments:

```python
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--preprocess_mode", default=2, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--learning_rate", default=1e-3, type=float)
parser.add_argument("--cosine_decay_steps", default=1000, type=int)
parser.add_argument("--cosine_decay_t_mul", default=2.0, type=float)
parser.add_argument("--cosine_decay_m_mul", default=1.0, type=float)
parser.add_argument("--cosine_decay_alpha", default=0.0, type=float)
parser.add_argument("--fit_verbose", default=1, type=int)
parser.add_argument("--name", default="resnet", type=str)
```

### Fit a PCA

```
python src/pca/fit_pca.py
```

No command line arguments can be passed.

### Predicting Similarity Judgments

```
python src/predicting/evaluate.py PREDICTOR
```

Argument parser:

```python
parser.add_argument("predictor", metavar="PREDICTOR", type=str)

parser.add_argument("--dist", type=str, default="mse")
parser.add_argument("--disable_tqdm", type=int, default=0)
parser.add_argument("--hps_path", type=str)
parser.add_argument("--data_sigma", type=float, default=1.0)
parser.add_argument("--emb_name", type=str, default="pca_emb_100")
parser.add_argument("--layer_to_use", type=int, default=-1)
```

`PREDICTOR` is a mandatory argument. You can set one of the following values

- `"simple"`
- `"random"`
- `"glow_simple"`
- `"glow_bayes"`
- `"glow_weighted"`
- `"glow_half"`
- `"pca"`
- `"resnet"`

You have to set the path to the `hyperparameters.json` file with `--hps_path` when using a trained model for predicting. You can set the distance metric to use (either `mse` or `cos`)  with `--dist`.

Results of our models are stored as CSVs at `src/predicting/out`.

### Other Visualizations

This folder contains some notebooks that were used to create visualizations.

- `compression.ipynb` visualizes compression of a trained Glow
- `interpolation.ipynb` visualizes latent interpolation of a trained Glow
- `learning_progress` visualizes loss curves for Glow and ResNet models
- `main_results` visualizes prediction performance of all the models

