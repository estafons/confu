import torch
import pytorch_lightning as pl
from src.modules.encoders.transformer_model import Transformer
from src.modules.models.confu import ConFu
from src.datasets.affect import AffectDataModule
from src.utils.evaluation import write_results_to_file, recall_at_k
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from src.utils.log_reg import evaluate_linear_probe

def get_loggers(cfg):

    name = f"{cfg.dataset.train_dataset}_{cfg.scenario}"
    loggers = [
        pl.loggers.TensorBoardLogger(
            save_dir="logs",
            name=name,
        )
    ]

    return loggers

@hydra.main(config_path="../../../configs", config_name="multibench")
def main(cfg):
    loggers = get_loggers(cfg)
    if cfg.scenario == "confu":
        return run_fusion_clip2(cfg, loggers)
    elif cfg.scenario == "triclip":
        run_baseline_clip2(cfg, loggers)
    elif cfg.scenario == "symile_baseline":
        run_symile_baseline(cfg, loggers)
    elif cfg.scenario == "triangle":
        run_triangle_baseline(cfg, loggers)
    elif cfg.scenario == "gram":
        run_gram_baseline(cfg, loggers)
    else:
        raise ValueError(f"Unknown scenario: {cfg.scenario}")



def run_fusion_clip2(cfg, loggers):
    from src.modules.models.confu import extract_embeddings, ConFu

    # Encoders
    modality1_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim1, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality2_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim2, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality3_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim3, dim=cfg.dataset.embedding.transformer_hid_dim)


    model = ConFu(modality1_encoder, modality2_encoder, modality3_encoder, embed_dim=cfg.dataset.embedding.common_dim, lr=cfg.training.lr, lambda_=cfg.lambda_, mask_ratio=cfg.mask_ratio, fusion_hidden_dim=cfg.fusion_hidden_dim, weight_decay=cfg.weight_decay)

    # DataModule
    dm = AffectDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pickle_name=cfg.dataset.pickle_name,
        dataset_name=cfg.dataset.train_dataset,
        samples_order=[0,1,2] # Using all three modalities
    )

    # Checkpoint callback for best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # make sure val_loss is logged in your LightningModule
        filename='best_model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=loggers
    )

    # Train
    trainer.fit(model, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = ConFu.load_from_checkpoint(best_model_path)

    print("Model loaded successfully. Now extracting embeddings...")

    # Extract embeddings
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()

    femb12, femb13, femb23, emb1, emb2, emb3, train_labels = extract_embeddings(model, train_loader, device="cuda:0")
    femb12_t, femb13_t, femb23_t, test_emb1, test_emb2, test_emb3, test_labels = extract_embeddings(model, test_loader, device="cuda:0")
    femb12v, femb13v, femb23v, val_emb1, val_emb2, val_emb3, val_labels = extract_embeddings(model, val_loader, device="cuda:0")
    print("Extracted embeddings for train and test sets.")

    
    accuracy1 = evaluate_linear_probe(train_feats=emb1, train_labels=train_labels, val_feats=val_emb1, val_labels=val_labels, test_feats=test_emb1, test_labels=test_labels)
    accuracy2 = evaluate_linear_probe(train_feats=emb2, train_labels=train_labels, val_feats=val_emb2, val_labels=val_labels, test_feats=test_emb2, test_labels=test_labels)
    accuracy3 = evaluate_linear_probe(train_feats=emb3, train_labels=train_labels, val_feats=val_emb3, val_labels=val_labels, test_feats=test_emb3, test_labels=test_labels)
    accuracy12 = evaluate_linear_probe(train_feats=femb12, train_labels=train_labels, val_feats=femb12v, val_labels=val_labels, test_feats=femb12_t, test_labels=test_labels)
    accuracy13 = evaluate_linear_probe(train_feats=femb13, train_labels=train_labels, val_feats=femb13v, val_labels=val_labels, test_feats=femb13_t, test_labels=test_labels)
    accuracy23 = evaluate_linear_probe(train_feats=femb23, train_labels=train_labels, val_feats=femb23v, val_labels=val_labels, test_feats=femb23_t, test_labels=test_labels)
    accuracy12_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb2], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb2], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb2], dim=-1), test_labels=test_labels)
    accuracy13_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb3], dim=-1), test_labels=test_labels)
    accuracy23_concat = evaluate_linear_probe(train_feats=torch.cat([emb2, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb2, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb2, test_emb3], dim=-1), test_labels=test_labels)

    # # Concatenate all modalities
    all_train = torch.cat([emb1, emb2, emb3], dim=-1)
    all_test = torch.cat([test_emb1, test_emb2, test_emb3], dim=-1)
    all_val = torch.cat([val_emb1, val_emb2, val_emb3], dim=-1)
    accuracy_all_test = evaluate_linear_probe(train_feats=all_train, train_labels=train_labels, val_feats=all_val, val_labels=val_labels, test_feats=all_test, test_labels=test_labels)
    
    val_split = int(0.7 * len(train_labels))
    train_split1 = all_train[val_split:]
    val_split1 = all_val[:val_split]
    val_labels_split1 = val_labels[:val_split]
    train_labels_split1 = train_labels[val_split:]

    print(torch.allclose(femb13_t, test_emb2))
    print((femb13_t - test_emb2).abs().mean())

    accuracy_all_val = evaluate_linear_probe(train_feats=train_split1, train_labels=train_labels_split1, val_feats=val_split1, val_labels=val_labels_split1, test_feats=all_val, test_labels=val_labels)
   # effective_rank_combined = effective_rank(all_test.numpy())
    # modality 1,2
    recall_at_k_dict_12 = recall_at_k(test_emb1, test_emb2, ks=[1, 5, 10], modalities=[1,2])
    recall_at_k_dict_12_val = recall_at_k(val_emb1, val_emb2, ks=[1, 5, 10], modalities=[1,2])
    # modality 1,3
    recall_at_k_dict_13 = recall_at_k(test_emb1, test_emb3, ks=[1, 5, 10], modalities=[1,3])
    recall_at_k_dict_13_val = recall_at_k(val_emb1, val_emb3, ks=[1, 5, 10], modalities=[1,3])
    # modality 2,3
    recall_at_k_dict_23 = recall_at_k(test_emb2, test_emb3, ks=[1, 5, 10], modalities=[2,3])
    recall_at_k_dict_23_val = recall_at_k(val_emb2, val_emb3, ks=[1, 5, 10], modalities=[2,3])

    recall_at_k_dict_12_3 = recall_at_k(femb12_t, test_emb3, ks=[1, 5, 10], modalities=[12,3])
    recall_at_k_dict_13_2 = recall_at_k(femb13_t, test_emb2, ks=[1, 5, 10], modalities=[13,2])
    recall_at_k_dict_23_1 = recall_at_k(femb23_t, test_emb1, ks=[1, 5, 10], modalities=[23,1])

    if cfg.results_on_test:
        recall_at_1_sum = recall_at_k_dict_12['M1->_M2_recall@1'] + recall_at_k_dict_13['M1->_M3_recall@1'] + recall_at_k_dict_23['M2->_M3_recall@1'] + recall_at_k_dict_12['M2->_M1_recall@1'] + recall_at_k_dict_13['M3->_M1_recall@1'] + recall_at_k_dict_23['M3->_M2_recall@1']
        recall_at_5_sum = recall_at_k_dict_12['M1->_M2_recall@5'] + recall_at_k_dict_13['M1->_M3_recall@5'] + recall_at_k_dict_23['M2->_M3_recall@5'] + recall_at_k_dict_12['M2->_M1_recall@5'] + recall_at_k_dict_13['M3->_M1_recall@5'] + recall_at_k_dict_23['M3->_M2_recall@5']
        recall_at_10_sum = recall_at_k_dict_12['M1->_M2_recall@10'] + recall_at_k_dict_13['M1->_M3_recall@10'] + recall_at_k_dict_23['M2->_M3_recall@10'] + recall_at_k_dict_12['M2->_M1_recall@10'] + recall_at_k_dict_13['M3->_M1_recall@10'] + recall_at_k_dict_23['M3->_M2_recall@10']
        accuracy_all = accuracy_all_test
    else:
        recall_at_1_sum = recall_at_k_dict_12_val['M1->_M2_recall@1'] + recall_at_k_dict_13_val['M1->_M3_recall@1'] + recall_at_k_dict_23_val['M2->_M3_recall@1'] + recall_at_k_dict_12_val['M2->_M1_recall@1'] + recall_at_k_dict_13_val['M3->_M1_recall@1'] + recall_at_k_dict_23_val['M3->_M2_recall@1']
        recall_at_5_sum = recall_at_k_dict_12_val['M1->_M2_recall@5'] + recall_at_k_dict_13_val['M1->_M3_recall@5'] + recall_at_k_dict_23_val['M2->_M3_recall@5'] + recall_at_k_dict_12_val['M2->_M1_recall@5'] + recall_at_k_dict_13_val['M3->_M1_recall@5'] + recall_at_k_dict_23_val['M3->_M2_recall@5']
        recall_at_10_sum = recall_at_k_dict_12_val['M1->_M2_recall@10'] + recall_at_k_dict_13_val['M1->_M3_recall@10'] + recall_at_k_dict_23_val['M2->_M3_recall@10'] + recall_at_k_dict_12_val['M2->_M1_recall@10'] + recall_at_k_dict_13_val['M3->_M1_recall@10'] + recall_at_k_dict_23_val['M3->_M2_recall@10']
        accuracy_all = accuracy_all_val


    results = [
        {"iteration": cfg.iteration, 
         "Modality1": accuracy1, 
         "Modality2": accuracy2, 
         "Modality3": accuracy3,
         **recall_at_k_dict_12, 
         **recall_at_k_dict_13, 
         **recall_at_k_dict_23,
         "Modality12": accuracy12,
         "Modality13": accuracy13,
         "Modality23": accuracy23,
         **recall_at_k_dict_12_3,
         **recall_at_k_dict_13_2,
         **recall_at_k_dict_23_1,
         "AllModalities": accuracy_all,
            "Modality12_concat": accuracy12_concat,
            "Modality13_concat": accuracy13_concat,
            "Modality23_concat": accuracy23_concat,
        },
    ]
    if cfg.iteration != 'not':
        write_results_to_file(results, cfg)

    recall_sum = recall_at_1_sum + recall_at_5_sum + recall_at_10_sum

    return accuracy_all, recall_sum

def run_symile_baseline(cfg, loggers):
    from src.modules.models.symile import SymileBaselineCLIPModule, extract_embeddings

    samples_order = cfg.dataset.samples_order
    if len(samples_order) != 3:
        raise ValueError(f"Expected 3 modalities, but got {len(samples_order)}. Please check your configuration.")

    # Encoders
    modality1_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim1, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality2_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim2, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality3_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim3, dim=cfg.dataset.embedding.transformer_hid_dim)

    model = SymileBaselineCLIPModule(modality1_encoder, modality2_encoder, modality3_encoder, embed_dim=cfg.dataset.embedding.common_dim, lr=cfg.training.lr)

    # DataModule
    dm = AffectDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pickle_name=cfg.dataset.pickle_name,
        dataset_name=cfg.dataset.train_dataset,
        samples_order=[0,1,2] # Using all three modalities
    )

    # Checkpoint callback for best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # make sure val_loss is logged in your LightningModule
        filename='best_model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=loggers,
    )

    # Train
    trainer.fit(model, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = SymileBaselineCLIPModule.load_from_checkpoint(best_model_path)

    print("Model loaded successfully. Now extracting embeddings...")

    # Extract embeddings
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    emb1, emb2, emb3, train_labels = extract_embeddings(model, train_loader, device="cuda:0")
    val_emb1, val_emb2, val_emb3, val_labels = extract_embeddings(model, val_loader, device="cuda:0")
    test_emb1, test_emb2, test_emb3, test_labels = extract_embeddings(model, test_loader, device="cuda:0")
    print("Extracted embeddings for train and test sets.")

    # Evaluate logistic regression
    accuracy1 = evaluate_linear_probe(train_feats=emb1, train_labels=train_labels, val_feats=val_emb1, val_labels=val_labels, test_feats=test_emb1, test_labels=test_labels)
    accuracy2 = evaluate_linear_probe(train_feats=emb2, train_labels=train_labels, val_feats=val_emb2, val_labels=val_labels, test_feats=test_emb2, test_labels=test_labels)
    accuracy3 = evaluate_linear_probe(train_feats=emb3, train_labels=train_labels, val_feats=val_emb3, val_labels=val_labels, test_feats=test_emb3, test_labels=test_labels)
    accuracy12_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb2], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb2], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb2], dim=-1), test_labels=test_labels)
    accuracy13_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb3], dim=-1), test_labels=test_labels)
    accuracy23_concat = evaluate_linear_probe(train_feats=torch.cat([emb2, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb2, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb2, test_emb3], dim=-1), test_labels=test_labels)
    
    all_train = torch.cat([emb1, emb2, emb3], dim=-1)
    all_test = torch.cat([test_emb1, test_emb2, test_emb3], dim=-1)
    all_val = torch.cat([val_emb1, val_emb2, val_emb3], dim=-1)
    accuracy_all = evaluate_linear_probe(train_feats=all_train, train_labels=train_labels, val_feats=all_val, val_labels=val_labels, test_feats=all_test, test_labels=test_labels)
    print('running recall @ k')

    # modality 1 from 2,3
    # modality 1,2
    recall_at_k_dict_12 = recall_at_k(test_emb1, test_emb2, ks=[1, 5, 10], modalities=[1,2])

    # modality 1,3
    recall_at_k_dict_13 = recall_at_k(test_emb1, test_emb3, ks=[1, 5, 10], modalities=[1,3])
    
    # modality 2,3
    recall_at_k_dict_23 = recall_at_k(test_emb2, test_emb3, ks=[1, 5, 10], modalities=[2,3])
    

    recall_at_k_dict_1_23 = model.recall_at_k(test_emb1, test_emb2, test_emb3, ks=[1, 5, 10], modalities=[1,23])
    # modality 2 from 1,3
    recall_at_k_dict_2_13 = model.recall_at_k(test_emb2, test_emb3, test_emb1, ks=[1, 5, 10], modalities=[2,13])
    # modality 3 from 1,2
    recall_at_k_dict_3_12 = model.recall_at_k(test_emb3, test_emb1, test_emb2, ks=[1, 5, 10], modalities=[3,12])

    print('finished recall at k')

    # Prepare results

    results = [
        {"iteration": cfg.iteration, 
         "Modality1": accuracy1, 
         "Modality2": accuracy2, 
         "Modality3": accuracy3,
         **recall_at_k_dict_12, 
         **recall_at_k_dict_13, 
         **recall_at_k_dict_23,
         **recall_at_k_dict_1_23,
         **recall_at_k_dict_2_13,
         **recall_at_k_dict_3_12,
         "AllModalities": accuracy_all,
            "Modality12_concat": accuracy12_concat,
            "Modality13_concat": accuracy13_concat,
            "Modality23_concat": accuracy23_concat,
        },
    ]
    write_results_to_file(results, cfg)


def run_triangle_baseline(cfg, loggers):
    from src.modules.models.triangle import TiangleBaselineCLIPModule, extract_embeddings

    samples_order = cfg.dataset.samples_order
    if len(samples_order) != 3:
        raise ValueError(f"Expected 3 modalities, but got {len(samples_order)}. Please check your configuration.")

    # Encoders
    modality1_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim1, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality2_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim2, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality3_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim3, dim=cfg.dataset.embedding.transformer_hid_dim)

    model = TiangleBaselineCLIPModule(modality1_encoder, modality2_encoder, modality3_encoder, embed_dim=cfg.dataset.embedding.common_dim, lr=cfg.training.lr)

    # DataModule
    dm = AffectDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pickle_name=cfg.dataset.pickle_name,
        dataset_name=cfg.dataset.train_dataset,
        samples_order=[0,1,2] # Using all three modalities
    )

    # Checkpoint callback for best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # make sure val_loss is logged in your LightningModule
        filename='best_model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=loggers,
    )

    # Train
    trainer.fit(model, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = TiangleBaselineCLIPModule.load_from_checkpoint(best_model_path)

    print("Model loaded successfully. Now extracting embeddings...")

    # Extract embeddings
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    emb1, emb2, emb3, train_labels = extract_embeddings(model, train_loader, device="cuda:0")
    val_emb1, val_emb2, val_emb3, val_labels = extract_embeddings(model, val_loader, device="cuda:0")
    test_emb1, test_emb2, test_emb3, test_labels = extract_embeddings(model, test_loader, device="cuda:0")
    print("Extracted embeddings for train and test sets.")

    # Evaluate logistic regression
    accuracy1 = evaluate_linear_probe(train_feats=emb1, train_labels=train_labels, val_feats=val_emb1, val_labels=val_labels, test_feats=test_emb1, test_labels=test_labels)
    accuracy2 = evaluate_linear_probe(train_feats=emb2, train_labels=train_labels, val_feats=val_emb2, val_labels=val_labels, test_feats=test_emb2, test_labels=test_labels)
    accuracy3 = evaluate_linear_probe(train_feats=emb3, train_labels=train_labels, val_feats=val_emb3, val_labels=val_labels, test_feats=test_emb3, test_labels=test_labels)
    accuracy12_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb2], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb2], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb2], dim=-1), test_labels=test_labels)
    accuracy13_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb3], dim=-1), test_labels=test_labels)
    accuracy23_concat = evaluate_linear_probe(train_feats=torch.cat([emb2, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb2, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb2, test_emb3], dim=-1), test_labels=test_labels)
    
    all_train = torch.cat([emb1, emb2, emb3], dim=-1)
    all_test = torch.cat([test_emb1, test_emb2, test_emb3], dim=-1)
    all_val = torch.cat([val_emb1, val_emb2, val_emb3], dim=-1)
    accuracy_all = evaluate_linear_probe(train_feats=all_train, train_labels=train_labels, val_feats=all_val, val_labels=val_labels, test_feats=all_test, test_labels=test_labels)
    print('running recall @ k')

    
    # modality 1 from 2,3
    # modality 1,2
    recall_at_k_dict_12 = recall_at_k(test_emb1, test_emb2, ks=[1, 5, 10], modalities=[1,2])

    # modality 1,3
    recall_at_k_dict_13 = recall_at_k(test_emb1, test_emb3, ks=[1, 5, 10], modalities=[1,3])
    
    # modality 2,3
    recall_at_k_dict_23 = recall_at_k(test_emb2, test_emb3, ks=[1, 5, 10], modalities=[2,3])
    

    recall_at_k_dict_1_23 = model.recall_at_k(test_emb1, test_emb2, test_emb3, ks=[1, 5, 10], modalities=[1,23])
    # modality 2 from 1,3
    recall_at_k_dict_2_13 = model.recall_at_k(test_emb2, test_emb3, test_emb1, ks=[1, 5, 10], modalities=[2,13])
    # modality 3 from 1,2
    recall_at_k_dict_3_12 = model.recall_at_k(test_emb3, test_emb1, test_emb2, ks=[1, 5, 10], modalities=[3,12])

    print('finished recall at k')

    # Prepare results

    results = [
        {"iteration": cfg.iteration, 
         "Modality1": accuracy1, 
         "Modality2": accuracy2, 
         "Modality3": accuracy3,
         **recall_at_k_dict_12, 
         **recall_at_k_dict_13, 
         **recall_at_k_dict_23,
         **recall_at_k_dict_1_23,
         **recall_at_k_dict_2_13,
         **recall_at_k_dict_3_12,
         "AllModalities": accuracy_all,
            "Modality12_concat": accuracy12_concat,
            "Modality13_concat": accuracy13_concat,
            "Modality23_concat": accuracy23_concat,
        },
    ]
    write_results_to_file(results, cfg)


def run_gram_baseline(cfg, loggers):
    from src.modules.models.gram import GramBaselineCLIPModule, extract_embeddings

    samples_order = cfg.dataset.samples_order
    if len(samples_order) != 3:
        raise ValueError(f"Expected 3 modalities, but got {len(samples_order)}. Please check your configuration.")

    # Encoders
    modality1_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim1, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality2_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim2, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality3_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim3, dim=cfg.dataset.embedding.transformer_hid_dim)

    model = GramBaselineCLIPModule(modality1_encoder, modality2_encoder, modality3_encoder, embed_dim=cfg.dataset.embedding.common_dim, lr=cfg.training.lr)

    # DataModule
    dm = AffectDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pickle_name=cfg.dataset.pickle_name,
        dataset_name=cfg.dataset.train_dataset,
        samples_order=[0,1,2] # Using all three modalities
    )

    # Checkpoint callback for best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # make sure val_loss is logged in your LightningModule
        filename='best_model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=loggers,
    )

    # Train
    trainer.fit(model, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = GramBaselineCLIPModule.load_from_checkpoint(best_model_path)

    print("Model loaded successfully. Now extracting embeddings...")

    # Extract embeddings
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    emb1, emb2, emb3, train_labels = extract_embeddings(model, train_loader, device="cuda:0")
    val_emb1, val_emb2, val_emb3, val_labels = extract_embeddings(model, val_loader, device="cuda:0")
    test_emb1, test_emb2, test_emb3, test_labels = extract_embeddings(model, test_loader, device="cuda:0")
    print("Extracted embeddings for train and test sets.")

    # Evaluate logistic regression
    accuracy1 = evaluate_linear_probe(train_feats=emb1, train_labels=train_labels, val_feats=val_emb1, val_labels=val_labels, test_feats=test_emb1, test_labels=test_labels)
    accuracy2 = evaluate_linear_probe(train_feats=emb2, train_labels=train_labels, val_feats=val_emb2, val_labels=val_labels, test_feats=test_emb2, test_labels=test_labels)
    accuracy3 = evaluate_linear_probe(train_feats=emb3, train_labels=train_labels, val_feats=val_emb3, val_labels=val_labels, test_feats=test_emb3, test_labels=test_labels)
    accuracy12_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb2], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb2], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb2], dim=-1), test_labels=test_labels)
    accuracy13_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb3], dim=-1), test_labels=test_labels)
    accuracy23_concat = evaluate_linear_probe(train_feats=torch.cat([emb2, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb2, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb2, test_emb3], dim=-1), test_labels=test_labels)
    
    all_train = torch.cat([emb1, emb2, emb3], dim=-1)
    all_test = torch.cat([test_emb1, test_emb2, test_emb3], dim=-1)
    all_val = torch.cat([val_emb1, val_emb2, val_emb3], dim=-1)
    accuracy_all = evaluate_linear_probe(train_feats=all_train, train_labels=train_labels, val_feats=all_val, val_labels=val_labels, test_feats=all_test, test_labels=test_labels)
    print('running recall @ k')

    
    # modality 1 from 2,3
    # modality 1,2
    recall_at_k_dict_12 = model.recall_at_k_2(test_emb1, test_emb2, ks=[1, 5, 10], modalities=[1,2])

    # modality 1,3
    recall_at_k_dict_13 = model.recall_at_k_2(test_emb1, test_emb3, ks=[1, 5, 10], modalities=[1,3])
    
    # modality 2,3
    recall_at_k_dict_23 = model.recall_at_k_2(test_emb2, test_emb3, ks=[1, 5, 10], modalities=[2,3])
    

    recall_at_k_dict_1_23 = model.recall_at_k(test_emb1, test_emb2, test_emb3, ks=[1, 5, 10], modalities=[1,23])
    # modality 2 from 1,3
    recall_at_k_dict_2_13 = model.recall_at_k(test_emb2, test_emb3, test_emb1, ks=[1, 5, 10], modalities=[2,13])
    # modality 3 from 1,2
    recall_at_k_dict_3_12 = model.recall_at_k(test_emb3, test_emb1, test_emb2, ks=[1, 5, 10], modalities=[3,12])

    print('finished recall at k')

    # Prepare results

    results = [
        {"iteration": cfg.iteration, 
         "Modality1": accuracy1, 
         "Modality2": accuracy2, 
         "Modality3": accuracy3,
         **recall_at_k_dict_12, 
         **recall_at_k_dict_13, 
         **recall_at_k_dict_23,
         **recall_at_k_dict_1_23,
         **recall_at_k_dict_2_13,
         **recall_at_k_dict_3_12,
         "AllModalities": accuracy_all,
            "Modality12_concat": accuracy12_concat,
            "Modality13_concat": accuracy13_concat,
            "Modality23_concat": accuracy23_concat,
        },
    ]
    write_results_to_file(results, cfg)

def run_baseline_clip2(cfg, loggers):
    from src.modules.models.triclip import ThreeModalityBaselineCLIPModule, extract_embeddings

    # samples_order = cfg.dataset.samples_order
    # if len(samples_order) != 3:
    #     raise ValueError(f"Expected 3 modalities, but got {len(samples_order)}. Please check your configuration.")

    # Encoders
    modality1_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim1, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality2_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim2, dim=cfg.dataset.embedding.transformer_hid_dim)
    modality3_encoder = Transformer(n_features=cfg.dataset.embedding.input_dim3, dim=cfg.dataset.embedding.transformer_hid_dim)



    model = ThreeModalityBaselineCLIPModule(modality1_encoder, modality2_encoder, modality3_encoder, embed_dim=cfg.dataset.embedding.common_dim, lr=cfg.training.lr)

    # DataModule
    dm = AffectDataModule(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pickle_name=cfg.dataset.pickle_name,
        dataset_name=cfg.dataset.train_dataset,
        samples_order=[0,1,2] # Using all three modalities
    )

    # Checkpoint callback for best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # make sure val_loss is logged in your LightningModule
        filename='best_model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=loggers,
    )

    # Train
    trainer.fit(model, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = ThreeModalityBaselineCLIPModule.load_from_checkpoint(best_model_path)

    print("Model loaded successfully. Now extracting embeddings...")

    # Extract embeddings
    dm.setup()
    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()
    val_loader = dm.val_dataloader()

    emb1, emb2, emb3, train_labels = extract_embeddings(model, train_loader, device="cuda:0")
    test_emb1, test_emb2, test_emb3, test_labels = extract_embeddings(model, test_loader, device="cuda:0")
    val_emb1, val_emb2, val_emb3, val_labels = extract_embeddings(model, val_loader, device="cuda:0")
    print("Extracted embeddings for train and test sets.")

    # Evaluate logistic regression
   # accuracy1 = train_evaluate_logistic(train_ia, train_labels, test_ia, test_labels, txt="Fused Modalities")
  #  accuracy2 = train_evaluate_logistic(train_txt, train_labels, test_txt, test_labels, txt="3rd Modality")
    # accuracy1 = train_evaluate_mlp(emb1, train_labels, test_emb1, test_labels, txt="Modality 1 ")
    # accuracy2 = train_evaluate_mlp(emb2, train_labels, test_emb2, test_labels, txt="Modality 2 ")
    # accuracy3 = train_evaluate_mlp(emb3, train_labels, test_emb3, test_labels, txt="Modality 3 ")
    accuracy1 = evaluate_linear_probe(train_feats=emb1, train_labels=train_labels, val_feats=val_emb1, val_labels=val_labels, test_feats=test_emb1, test_labels=test_labels)
    accuracy2 = evaluate_linear_probe(train_feats=emb2, train_labels=train_labels, val_feats=val_emb2, val_labels=val_labels, test_feats=test_emb2, test_labels=test_labels)
    accuracy3 = evaluate_linear_probe(train_feats=emb3, train_labels=train_labels, val_feats=val_emb3, val_labels=val_labels, test_feats=test_emb3, test_labels=test_labels)
    accuracy12_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb2], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb2], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb2], dim=-1), test_labels=test_labels)
    accuracy13_concat = evaluate_linear_probe(train_feats=torch.cat([emb1, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb1, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb1, test_emb3], dim=-1), test_labels=test_labels)
    accuracy23_concat = evaluate_linear_probe(train_feats=torch.cat([emb2, emb3], dim=-1), train_labels=train_labels, val_feats=torch.cat([val_emb2, val_emb3], dim=-1), val_labels=val_labels, test_feats=torch.cat([test_emb2, test_emb3], dim=-1), test_labels=test_labels)
    all_train = torch.cat([emb1, emb2, emb3], dim=-1)
    all_test = torch.cat([test_emb1, test_emb2, test_emb3], dim=-1)
    all_val = torch.cat([val_emb1, val_emb2, val_emb3], dim=-1)
    accuracy_all = evaluate_linear_probe(train_feats=all_train, train_labels=train_labels, val_feats=all_val, val_labels=val_labels, test_feats=all_test, test_labels=test_labels)


    # modality 1,2
    recall_at_k_dict_12 = recall_at_k(test_emb1, test_emb2, ks=[1, 5, 10], modalities=[1,2])
    # modality 1,3
    recall_at_k_dict_13 = recall_at_k(test_emb1, test_emb3, ks=[1, 5, 10], modalities=[1,3])
    # modality 2,3
    recall_at_k_dict_23 = recall_at_k(test_emb2, test_emb3, ks=[1, 5, 10], modalities=[2,3])

    # Prepare results

    results = [
        {"iteration": cfg.iteration, 
         "Modality1": accuracy1, 
         "Modality2": accuracy2, 
         "Modality3": accuracy3,
         **recall_at_k_dict_12, 
         **recall_at_k_dict_13, 
         **recall_at_k_dict_23,
            "AllModalities": accuracy_all,
                "Modality12_concat": accuracy12_concat,
                "Modality13_concat": accuracy13_concat,
                "Modality23_concat": accuracy23_concat,
        },
    ]
    write_results_to_file(results, cfg)




if __name__ == "__main__":
    main()
