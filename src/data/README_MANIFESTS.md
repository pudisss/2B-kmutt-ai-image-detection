# Using Manifest Files for Training and Evaluation

This directory contains manifest CSV files that allow you to have full control over which data samples are used for training and testing.

## Manifest Files

- **train_manifest.csv**: Samples used for training
- **test_manifest.csv**: Samples used for final evaluation

## Manifest Format

Each manifest CSV file should have the following columns:

```csv
filepath,split,label,dataset,architecture
cnndetection/train/1_fake/image.png,train,1,CNNDetection,progan
cnndetection/train/0_real/image.png,train,0,CNNDetection,natural
```

### Column Descriptions:

- **filepath**: Relative path to the image from data root directory
- **split**: Original split designation (train/test) - informational only
- **label**: 0 for real images, 1 for fake/AI-generated images
- **dataset**: Source dataset name (CNNDetection, DiffusionForensics, etc.)
- **architecture**: Generator architecture (progan, stylegan, ddpm, natural, etc.)

## Usage

### Training with Manifests

```bash
# Use default manifest locations (src/data/*.csv)
python scripts/train.py --use-manifests --data-dir /path/to/data

# Or specify custom manifest paths
python scripts/train.py \
    --train-manifest /path/to/custom_train.csv \
    --test-manifest /path/to/custom_test.csv \
    --data-dir /path/to/data
```

Validation is created automatically by stratified splitting of `train_manifest.csv`
(10% by default, controlled by `val_split` in the data module).

### Evaluation with Manifests

```bash
# Use default test manifest
python scripts/evaluate.py --use-manifests --per-architecture

# Or specify custom test manifest
python scripts/evaluate.py \
    --test-manifest /path/to/custom_test.csv \
    --per-architecture \
    --checkpoint checkpoints/best.pt
```

### Training without Manifests (Automatic Splitting)

```bash
# Uses config.yaml to define train/test model splits
python scripts/train.py --data-dir /path/to/data
```

## Benefits of Using Manifests

1. **Full Control**: You decide exactly which samples go into train/test
2. **Reproducibility**: Same manifest = same data split every time
3. **Custom Splits**: Create splits based on your research needs
4. **Easy Modifications**: Edit CSV files to adjust dataset composition
5. **Version Control**: Track changes to data splits in git
6. **Experiment Tracking**: Different manifests for different experiments

## Creating Your Own Manifests

You can create custom manifest files by:

1. **Filtering existing manifests**: Edit the provided CSV files
2. **Combining datasets**: Merge multiple metadata CSVs
3. **Custom sampling**: Use pandas to create balanced/stratified splits

Example Python code to create custom manifests:

```python
import pandas as pd

# Load original metadata
df = pd.read_csv('/path/to/metadata.csv')

# Filter for specific architectures
train_df = df[df['architecture'].isin(['progan', 'stylegan', 'ddpm'])]
test_df = df[df['architecture'].isin(['stylegan2', 'midjourney'])]

# Save as manifests
train_df.to_csv('custom_train_manifest.csv', index=False)
test_df.to_csv('custom_test_manifest.csv', index=False)
```

## Notes

- Manifests take priority over automatic splitting when provided
- File paths in manifests should be relative to `--data-dir`
- The `split` column is informational; actual usage is determined by which manifest file contains the sample
- Validation is always created by splitting the training manifest (10% by default)

## Current Manifest Statistics

Check the number of samples in each manifest:

```bash
wc -l src/data/train_manifest.csv src/data/test_manifest.csv
```

View sample counts by architecture and label:

```python
import pandas as pd

df = pd.read_csv('src/data/train_manifest.csv')
print(df.groupby(['architecture', 'label']).size())
```
