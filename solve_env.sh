#! /bin/bash
export http_proxy=http://oversea-squid4.sgp.txyun:11080 
export https_proxy=http://oversea-squid4.sgp.txyun:11080

apt install git-lfs
mkdir model
cd model
git lfs install
git clone https://huggingface.co/bert-base-uncased
git clone https://huggingface.co/bert-large-uncased
cd ..

conda config --add envs_dirs /mmu_nlp/wuxing/maguangyuan/miniconda3/envs
conda config --add pkgs_dirs /mmu_nlp/wuxing/maguangyuan/miniconda3/pkgs

conda create -n NLP_RE python=3.8
conda init bash
sleep 2s
source ~/.bashrc
sleep 2s
conda activate NLP_RE
sleep 2s
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install -r Requirements.txt