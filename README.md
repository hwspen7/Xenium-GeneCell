# Xenium-GeneCell

## Windows (NVIDIA GPU)

**1. Clone Repository**

```
git clone https://github.com/YOUR_USERNAME/Xenium-GeneCell.git
cd Xenium-GeneCell
```

**2. Create Conda Environment**

```
conda env create -f environment.yml
conda activate cellmae
```

**3. Install PyTorch**

```
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

**4. Install Dependencies**

```
python -m pip install -e ".[dev,notebook]"
```

## macOS (Apple Silicon)

**1. Clone Repository**

```
git clone https://github.com/YOUR_USERNAME/Xenium-GeneCell.git
cd Xenium-GeneCell
```

**2. Create Conda Environment**

```
conda env create -f environment.yml
conda activate cellmae
```

**3. Install PyTorch**

```
python -m pip install torch==2.13.0
```

**4. Install Dependencies**

```
python -m pip install -e ".[dev,notebook]"
```

## Verification

```
python -m pip check
cellmae doctor --compute-test
python -m pytest -q
```
