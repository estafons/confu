# ConFu Instructions

## Illustration of method

For an illustration of our method see file src/modules/models/confu.py


## data proposed structure
```
src/
data/
└── avmnist
    ├── audio
    │   ├── test_data.npy
    │   └── train_data.npy
    ├── image
    │   ├── test_data.npy
    │   └── train_data.npy
    ├── test_labels.npy
    └── train_labels.npy
└── multibench
    ├── humor.pkl
    ├── im.pkl
    ├── mosei_senti_data.pkl
    ├── mosi_data.pkl
    └── sarcasm.pkl
└── birds_mml_csvs
    ├── taxa_ssw60_vb100_all.csv
    ├── taxa.csv
└── CUB_200_2011
    ├── ..
└── birds_mml_csvs
    ├── taxa_ssw60_vb100_all.csv
    ├── taxa.csv
└── ssw60
    ├── audio_ml
    │   ├── ..
    ├── images_inat
    │   ├── ..
    ├── images_nabirds
    │   ├── ..
    ├── video_ml
    │   ├── ..
    ├── audio_ml.csv
    ├── images_inat.csv
    ├── images_nabirds.csv
    ├── video_ml.csv
    └── ..
└── vb100
    ├── vb100_all_videos
    │   ├── ..
    ├── vb100_annotations_v1
    │   ├── ..
    ├── vb100_audio
    │   ├── ..
    ├── vb100_taxonomy
    │   ├── ..
```



## Running MULTIBENCH experiments

Each experiment was repeated for **5 runs**, and the **mean** and **standard deviation** were computed.

1. **Install the required dependencies.**
2. **Download the dataset** from the MultiBench repository:  
   https://github.com/pliang279/MultiBench
3. **Set the `results_path`** in your Hydra configuration (`config.yaml`).  
   This determines where intermediate CSV results will be stored.  
   Example:
   ```yaml
   results_path: ${oc.env:HOME}/contrastive_fusion/results/
4. **Run the script**
```
bash scripts/multibench.sh
```


## Running AV-MNIST experiments

Experiments were repeated for **5 runs**, and the **mean** and **standard deviation** were computed.

1. **Download the dataset:**  
   https://github.com/pliang279/MultiBench?tab=readme-ov-file#multimedia
2. **Configure the results and data paths** in the Hydra config (`av_mnist.yaml`).
3. **Run the experiment script:**
   ```bash
   bash scripts/av_mnist.sh
    ```

## Running Birds-MML experiments

Training from scratch requires approximately **1–2 days** on a single GPU.

1. **Install the required dependencies.**
2. **Download the evaluation datasets:**  
   - SSW60: https://github.com/visipedia/ssw60  
   - VB100: https://arma.sourceforge.net/vb100/
3. **Download a pretrained model**, or download the training data and **train the model** yourself.
4. **Configure the data and model paths** in `configs/birds.yaml`.
5. **Run the training and evaluation scripts:**
   ```bash
   bash scripts/birds_mml_train.sh
   bash scripts/bird_zero_shot.sh
   bash scripts/birds_few_shot.sh
   ```

