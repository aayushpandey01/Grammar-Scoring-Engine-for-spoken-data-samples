# Grammar-Scoring-Engine-for-spoken-data-samples
Predicts a numeric score for each spoken audio clip from the Kaggle SHL Hiring Assessment competition. The approach combines handcrafted acoustic and fluency features with frozen wav2vec2 speech embeddings, and blends several regressors into one ensemble.

# Approach
Handcrafted features (librosa, 16 kHz, first 40 s of each clip)
MFCCs (30) with delta and delta-delta, mel-band statistics, spectral contrast, centroid, rolloff, bandwidth, flatness, zero-crossing rate and RMS (mean and std of each)
Fluency proxies: speech duration and ratio, number of segments, pause count and length, onset rate
Speech embeddings (facebook/wav2vec2-base, frozen, first 20 s of each clip)
Mean and std pooled hidden states from layers 6, 9 and 12
Runs on GPU if available; loads a locally attached model first, so it also works with Internet off
Models: Ridge, RBF-SVR and gradient boosting on the handcrafted features; Ridge and SVR on the embeddings; Ridge on the combined features
Ensemble: non-negative linear blend of out-of-fold predictions (5-fold CV), predictions clipped to the training label range
Submission: one row per clip in test.csv, with column names taken from sample_submission.csv

# Results (5 fold cross validation on the training set)
Model	Pearson	RMSE
Handcrafted, SVR	0.790	0.759
wav2vec2, Ridge	0.843	0.667
wav2vec2, SVR	0.846	0.662
Handcrafted + wav2vec2, Ridge	0.851	0.651
Ensemble	add your CV value	add your CV value

# Notes
The competition's sample_submission.csv is stale: it has fewer rows than test.csv. The scorer expects one row per test clip, so the notebook builds the submission from test.csv.
No external data is used and test labels are never read. Only the public pretrained wav2vec2 model is used, which the competition rules allow.
Random seed is fixed (42) for reproducibility.

# Reproduce
Create a Kaggle notebook and attach the competition data.
Import shl_notebook_final.ipynb.
Set the accelerator to a GPU (T4 or P100). Enable Internet, or attach a wav2vec2-base model through Add Input.
Run all cells. The last cell writes /kaggle/working/submission.csv.

To run without pretrained models, set USE_W2V = False in the settings cell (lower score, much faster).

# Requirements

numpy, pandas, scipy, scikit-learn, librosa, joblib, matplotlib, torch, transformers (all preinstalled on Kaggle).
