# Predicting Human Similarity Judgments Using Normalizing-Flows
Code for reproducing results in my thesis "Predicting Human Similarity Judgments Using Normalizing Flows" (Cognitive Science B.Sc.).

## Requirements

The `requirement.txt` for a complete list of our Python 3.9 environment.

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

Use `--help` to get information about optional command line arguments.

A pretrained model can be downloaded from [Hessenbox](https://hessenbox.tu-darmstadt.de/getlink/fiJRjP6ivPqSrwBGKmQvELCC/Predicting%20Human%20Similarity%20Judgments%20Using%20Normalizing%20Flows). Just put `128-2_2022-09-20_17:36:27` or `add-128-4_2022-10-03_01:11:41` into `src/glow/safe`.

### Train a ResNet

```
python src/resnet/train_resnet.py
```

Use `--help` to get information about optional command line arguments.

Pretrained models can be downloaded from [Hessenbox](https://hessenbox.tu-darmstadt.de/getlink/fiJRjP6ivPqSrwBGKmQvELCC/Predicting%20Human%20Similarity%20Judgments%20Using%20Normalizing%20Flows). Just put `resnet-D-2_2022-10-05_10:20:37` into `src/resnet/safe`.

### Fit a PCA

```
python src/pca/fit_pca.py
```

No command line arguments can be passed.

### Predicting Similarity Judgments

```
python src/predicting/evaluate.py PREDICTOR
```

Use `--help` to get information about mandatory and optional command line arguments.

Results of our models are stored as CSVs at `src/predicting/out`.

### Other Visualizations

The folder `src/visualizations` contains some notebooks that we used :

- `compression.ipynb` visualizes compression of a trained Glow
- `interpolation.ipynb` visualizes latent interpolation of a trained Glow
- `learning_progress` visualizes loss curves for Glow and ResNet models
- `main_results` visualizes prediction performance of all the models

