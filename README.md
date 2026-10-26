# 🩺 Breast Cancer Classifier (PyTorch)

A feedforward neural network built in PyTorch that classifies breast tumours as **malignant** or **benign** from 30 features computed from digitized images of fine needle aspirate (FNA) biopsies.

## Results

| Metric | Score |
|---|---|
| Test accuracy | **XX%** <!-- run the script and paste the printed accuracy --> |

<!-- Add the training loss curve and confusion matrix images here -->
<!-- ![Training loss](images/loss_curve.png) -->

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) from the UCI Machine Learning Repository, loaded automatically with `ucimlrepo`.

- **569 samples**, 30 numeric features (radius, texture, perimeter, area, smoothness, concavity, symmetry, and more, as mean, standard error and worst value)
- **Labels:** Malignant (1) / Benign (0)

## Approach

1. **Preprocessing:** labels encoded (M → 1, B → 0), features standardized with `StandardScaler`
2. **Split:** stratified 80/20 train-test split, so both sets keep the same benign/malignant ratio
3. **Model:** fully connected network

   ```
   Input (30) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(16) → ReLU → Linear(1) → Sigmoid
   ```

4. **Training:** binary cross-entropy loss, Adam optimizer (lr = 0.001), 100 epochs
5. **Evaluation:** accuracy, confusion matrix, and a per-class classification report (precision, recall, F1)

In a medical setting, **recall on the malignant class** matters most, because a missed cancer is far more costly than a false alarm. The classification report shows this directly.

## Run it

```bash
pip install -r requirements.txt
python "Breast _Cancer _predictor.py"
```

The script prints the training loss every 10 epochs, shows the loss curve, and prints the test accuracy, confusion matrix and classification report.

## Tech stack

Python · PyTorch · scikit-learn · pandas · NumPy · Matplotlib

## Possible improvements

- Compare against baselines (logistic regression, random forest)
- Add a validation set and early stopping
- Tune the decision threshold to prioritize recall on malignant cases
- Use k-fold cross-validation for a more stable accuracy estimate on this small dataset

---

Built by [Yashneil Purkayastha](https://www.linkedin.com/in/yashneil-purkayastha-87a593371)
