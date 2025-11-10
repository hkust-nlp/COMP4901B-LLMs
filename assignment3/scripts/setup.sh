pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers[torch]==4.57.1
pip install vllm==0.10.2 

pip install  deepspeed --no-deps
pip install  hjson
pip install ninja
pip install absl-py langdetect nltk immutabledict
pip install datasets==4.0.0
pip install wandb
pip install peft
# Download resources 
# python -c "import nltk; nltk.download('punkt_tab')"
# wget https://huggingface.co/datasets/PeterV09/smol-smoltalk-6k/resolve/main/smol-smoltalk-6k.json
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-0.6B', repo_type='model', local_dir='Qwen3-0.6B', cache_dir='.cache')"

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='gsm8k', repo_type='dataset', local_dir='gsm8k-local', cache_dir='.cache')"
