from src.modules.encoders.resnet import create_image_encoder, create_audio_encoder, create_text_encoder
from src.modules.models.confu import ConFu
from src.datasets.get_dataset_birds_eval import get_dataset
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import tqdm
import hydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_eval_confu(cfg, dataset_name='vb100', single_frame_eval=False, sample_sec=2.0):
    # --- Evaluate all checkpoints ---
    embed_dim = cfg.embed_dim
   
    # --- Create encoders ---
    image_encoder = create_image_encoder(embed_dim=embed_dim, in_channels=3, backbone='resnet50')
    audio_encoder = create_audio_encoder(embed_dim=embed_dim, in_channels=1, backbone='resnet50')
    text_encoder = create_text_encoder(embed_dim=embed_dim)

    dataset = get_dataset(cfg, dataset_name, single_frame_eval=single_frame_eval, sample_sec=sample_sec)

    
    ckpt_path = cfg.confu_chkpt_path

   
    texts = dataset.get_texts()
    num_texts = len(texts)
    dummy_audios = torch.randn(num_texts, 1, 128, 431).to(device)  # dummy audio
    dummy_images = torch.randn(num_texts, 3, 224, 224).to(device)  # dummy image

    # optionally precompute text embeddings with dummy inputs
    
    results = []

    model = ConFu.load_from_checkpoint(
        ckpt_path,
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder
    )
    
    model.eval()
    model.to(device)  # ensure on correct device
    with torch.no_grad():
        text_embs = model(dummy_images, dummy_audios, texts)[-1]
        text_embs.to(device)

# Loop over all checkpoints
    
    print(f"Evaluating checkpoint: {ckpt_path}")

    model.to(device)  # ensure on correct device

    correct = 0
    total = 0
    correct_image = 0
    correct_audio = 0

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    # correct labels are in diagonal
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
            images, audios, batch_texts, labels = batch
            images = images #.to(model.device)
            audios = audios.to(model.device)
            
            
            if dataset_name == 'cub' or single_frame_eval:
                images = images.to(model.device)
                image_audio, image_text, audio_text, image_emb, audio_emb, text_emb = model(
                    images, audios, batch_texts
                )
            else:
                image_audio, image_text, audio_text, image_emb, audio_emb, text_emb = model.video_forward(
                    images, audios, batch_texts
                )

            # compute similarities
            logits_per_image_audio = image_audio @ text_embs.t()
            preds = logits_per_image_audio.softmax(dim=-1).argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

            logits_per_image = image_emb @ text_embs.t()
            preds_image = logits_per_image.softmax(dim=-1).argmax(dim=-1)
            correct_image += (preds_image.cpu() == labels).sum().item()

            logits_per_audio = audio_emb @ text_embs.t()
            preds_audio = logits_per_audio.softmax(dim=-1).argmax(dim=-1)
            correct_audio += (preds_audio.cpu() == labels).sum().item()

    print(f"Checkpoint: {ckpt_path}")
    print(f"  Accuracy (image+audio): {correct / total:.4f}")
    print(f"  Accuracy (image only): {correct_image / total:.4f}")
    print(f"  Accuracy (audio only): {correct_audio / total:.4f}")
    results.append({
        "checkpoint": str(ckpt_path),
        "accuracy": correct / total,
        "accuracy_image": correct_image / total,
        "accuracy_audio": correct_audio / total
    })

    return results


def single_frame_eval_results_summary(cfg):
    results_summary = []

    # load_and_eval_trimodal_baseline(epoch, dataset_name)
    for sample_sec in np.arange(0.5, 10, 0.5):
        print(f"\n=== Evaluating with frame sample second: {sample_sec} ===")
        results_our = load_and_eval_confu(cfg, dataset_name=cfg.dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        # results_2_modal = load_and_eval_2_modal_baseline(epoch, dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        # results_trimodal = load_and_eval_trimodal_baseline(epoch, dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        # results_symile = load_and_eval_symile_baseline(epoch, dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        #results_triangle = load_and_eval_triangle_baseline(epoch, dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        # results_triangle = load_parquet_and_eval_triangle_baseline(evaluate_epoch=epoch, dataset_name=dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        #results_gram = load_parquet_and_eval_gram_baseline(evaluate_epoch=epoch, dataset_name=dataset_name, single_frame_eval=True, sample_sec=sample_sec)
        #results_gram = load_and_eval_gram_baseline(epoch, dataset_name, single_frame_eval=True, sample_sec=sample_sec)

        print(f" Our model results: {results_our}")
        #  print(f" 2-modal baseline results: {results_2_modal}")
        #  print(f" Trimodal results: {results_trimodal}")
        # print(f" Symile baseline results: {results_symile}")
        # print(f" Triangle baseline results: {results_triangle}")
        #  print(f" GRAM baseline results: {results_gram}")
        

        #aggregate so we can compute averages later
        results_summary.append({
            "sample_sec": sample_sec,
            "our_model": results_our,
            # "2_modal_baseline": results_2_modal,
            #  "trimodal": results_trimodal,
        #    "symile_baseline": results_symile,
        #    "triangle_baseline": results_triangle,
            #"gram_baseline": results_gram,
        })
    # save summary to file
    with open(f"zero_shot_single_frame_results_summary_{cfg.dataset_name}.json", "w") as f:
        json.dump(results_summary, f)
    print("\n=== Summary of Results Across Different Frames ===")
    # print results_summary
    print(results_summary)

@hydra.main(config_path="../../../configs", config_name="birds")
def main(cfg):
    if cfg.single_frame_eval is False:
        load_and_eval_confu(cfg, dataset_name=cfg.dataset_name, single_frame_eval=False)
    else:
        single_frame_eval_results_summary(cfg)


if __name__ == "__main__":
    main()