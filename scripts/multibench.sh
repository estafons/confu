### sarcasm

# confu
python3 -m src.experiments.multibench.main -m scenario=confu lambda_=0.5 mask_ratio=0.0 dataset=sarcasm dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False +fusion_hidden_dim=512

# gram
python3 -m src.experiments.multibench.main -m scenario=gram dataset=sarcasm dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# triangle
python3 -m src.experiments.multibench.main -m scenario=triangle dataset=sarcasm dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# symile
python3 -m src.experiments.multibench.main -m scenario=symile dataset=sarcasm dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# triclip
python3 -m src.experiments.multibench.main -m scenario=triclip dataset=sarcasm dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

### MOSI
# confu
python3 -m src.experiments.multibench.main -m scenario=confu lambda_=0.5 mask_ratio=0.0 dataset=mosi dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False +fusion_hidden_dim=512

# gram
python3 -m src.experiments.multibench.main -m scenario=gram dataset=mosi dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# triangle
python3 -m src.experiments.multibench.main -m scenario=triangle dataset=mosi dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# symile
python3 -m src.experiments.multibench.main -m scenario=symile dataset=mosi dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# triclip
python3 -m src.experiments.multibench.main -m scenario=triclip dataset=mosi dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

### HUMOR
# confu
python3 -m src.experiments.multibench.main -m scenario=confu lambda_=0.5 mask_ratio=0.0 dataset=humor dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False +fusion_hidden_dim=512

# gram
python3 -m src.experiments.multibench.main -m scenario=gram dataset=humor dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# triangle
python3 -m src.experiments.multibench.main -m scenario=triangle dataset=humor dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# symile
python3 -m src.experiments.multibench.main -m scenario=symile dataset=humor dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

# triclip
python3 -m src.experiments.multibench.main -m scenario=triclip dataset=humor dataset.embedding.common_dim=256 +iteration=1,2,3,4,5 +'dataset.samples_order=[0,1,2]' training.max_epochs=100 +results_on_test=True +hyperparameter_search=False

python3 -m src.utils.aggregate