import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
import random
import json
import hydra
from src.modules.encoders.resnet import create_audio_encoder, create_image_encoder, create_text_encoder
import torch
from src.datasets.get_dataset_birds_eval import get_dataset
from src.modules.models.confu import ConFu
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

def few_shot_split(df, n_shots, seed=None):
    """Split dataset into support and query sets."""
    if seed is not None:
        random.seed(seed)
    grouped = df.groupby("label")
    support_idx = []
    query_idx = []
    for label, group in grouped:
        idxs = group.index.tolist()
        random.shuffle(idxs)
        support_idx += idxs[:n_shots]
        query_idx += idxs[n_shots:]
    return support_idx, query_idx

def evaluate_few_shot_confu(model, dataset, n_shots_list=[1, 2, 5, 10], k_repeats=10, device="cuda", single_frame=False):
    """
    Few-shot evaluation loop.
    """
    results = defaultdict(dict)
    fused_features = []
    audio_features = []
    image_features = []
    all_labels = []

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, audios, texts, labels = batch
            images = images #.to(device)
            audios = audios.to(device)

            
            # You can choose which modality (or fusion)
            if dataset.name == "cub_birds" or single_frame:
                images = images.to(device)
                image_audio, image_text, audio_text, image_emb, audio_emb, text_emb = model(
                images, audios, texts
            )
            else:
                image_audio, image_text, audio_text, image_emb, audio_emb, text_emb = model.video_forward(
                    images, audios, texts
                )
            fused_features.append(image_audio.cpu())
            audio_features.append(audio_emb.cpu())
            image_features.append(image_emb.cpu())
            all_labels.append(labels)

    fused_features = torch.cat(fused_features, dim=0).numpy()
    audio_features = torch.cat(audio_features, dim=0).numpy()
    image_features = torch.cat(image_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    df = pd.DataFrame({"idx": np.arange(len(all_labels)), "label": all_labels})

    for features, name in zip(
        [fused_features, image_features, audio_features],
        ["Fused", "Image", "Audio"]
    ):
        print(f"\n--- Evaluating {name} features ---")
        for n_shots in n_shots_list:
            print(f"\n=== Few-Shot ({n_shots}-shot) Evaluation ===")
            accs = []
            for seed in range(k_repeats):
                support_idx, query_idx = few_shot_split(df, n_shots=n_shots, seed=seed)

                X_train = features[support_idx]
                y_train = all_labels[support_idx]
                X_test = features[query_idx]
                y_test = all_labels[query_idx]

                # Fit simple classifier
                clf = LogisticRegression(max_iter=500, multi_class="multinomial")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accs.append(acc)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"{name} features - {n_shots}-shot mean acc: {mean_acc:.4f} ± {std_acc:.4f}")
            results[name][n_shots] = {"mean": mean_acc, "std": std_acc}

    return results


def eval_confu_clip_multi_frame(cfg, dataset_name='ssw60'):
    embed_dim = cfg.embed_dim
    ckpt_path = cfg.confu_chkpt_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_encoder = create_image_encoder(embed_dim=embed_dim, in_channels=3, backbone='resnet50')
    audio_encoder = create_audio_encoder(embed_dim=embed_dim, in_channels=1, backbone='resnet50')
    text_encoder = create_text_encoder(embed_dim=embed_dim)
    model = ConFu.load_from_checkpoint(
        ckpt_path,
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder
    )
    model.eval().to(device)
    results_all = {}
    
        
    dataset = get_dataset(cfg, dataset_name, single_frame_eval=False)
    results = evaluate_few_shot_confu(model, dataset, n_shots_list=[1, 2, 5, 10, 20], k_repeats=10, device=device, single_frame=False)
    print(results)

    # save results to a json file
    with open(f"multi_frame_few_shot_results_confu_{dataset_name}.json", "w") as f:
        json.dump(results_all, f)

def eval_confu_clip_single_frame(cfg, dataset_name='ssw60'):
    embed_dim = cfg.embed_dim
    ckpt_path = cfg.confu_chkpt_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_encoder = create_image_encoder(embed_dim=embed_dim, in_channels=3, backbone='resnet50')
    audio_encoder = create_audio_encoder(embed_dim=embed_dim, in_channels=1, backbone='resnet50')
    text_encoder = create_text_encoder(embed_dim=embed_dim)
    model = ConFu.load_from_checkpoint(
        ckpt_path,
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder
    )
    model.eval().to(device)
    results_all = {}
    for sample_sec in np.arange(0.5, 10, 0.5):
        print(f'on iteration {sample_sec} of fusion')
        dataset = get_dataset(cfg, dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        results = evaluate_few_shot_confu(model, dataset, n_shots_list=[1, 2, 5, 10, 20], k_repeats=10, device=device, single_frame=True)
        results_all[sample_sec] = results
    print(results)

    # save results to a json file
    with open(f"single_frame_few_shot_results_confu_{dataset_name}.json", "w") as f:
        json.dump(results_all, f)


@hydra.main(config_path="../../../configs", config_name="birds")
def main(cfg):
    if cfg.single_frame_eval is False:
        eval_confu_clip_multi_frame(cfg, dataset_name=cfg.dataset_name)
    else:
        eval_confu_clip_single_frame(cfg, dataset_name=cfg.dataset_name)


if __name__ == "__main__":
    main()