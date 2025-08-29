#!/bin/bash
# Mac Pro 2013 Home Server Setup Script
# Optimized for computational experiments with HD computing, genomics, and AI

echo "🖥️  Mac Pro Home Server Setup - Computational Experiments Edition"
echo "================================================================"

# 1. Enable remote access and server features
echo "📡 Configuring remote access..."
sudo systemsetup -setremotelogin on
sudo systemsetup -setcomputersleep Never
sudo systemsetup -setdisplaysleep 180
sudo systemsetup -setharddisksleep Never

# 2. Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 3. Install essential server and development tools
echo "📦 Installing core packages..."
brew install \
    tmux \
    htop \
    wget \
    git \
    python@3.11 \
    node \
    rust \
    cmake \
    llvm \
    openblas \
    fftw \
    hdf5 \
    jupyter

# 4. Install computational libraries for HD computing experiments
echo "🧮 Installing computational libraries..."
brew install \
    eigen \
    armadillo \
    boost \
    tbb \
    gsl \
    numpy \
    scipy

# 5. Install bioinformatics tools for genomics experiments
echo "🧬 Installing bioinformatics tools..."
brew install \
    samtools \
    bcftools \
    bedtools \
    bwa \
    minimap2 \
    seqtk

# 6. Set up Python environment for experiments
echo "🐍 Setting up Python environment..."
python3 -m venv ~/experiment_env
source ~/experiment_env/bin/activate

pip install --upgrade pip
pip install \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    torch \
    tensorflow-macos \
    jax \
    numba \
    cython \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    tqdm \
    biopython \
    pysam \
    pyvcf

# 7. Install HD computing and efficiency-focused libraries
pip install \
    torch-hd \
    nengo \
    nengo-spa \
    gudhi \
    ripser \
    giotto-tda \
    hypertools

echo "✅ Basic setup complete!"
