git clone https://github.com/openai/CLIP
pip install -e ./CLIP

git clone https://github.com/crowsonkb/guided-diffusion
pip install -e ./guided-diffusion

pip install lpips datetime timm
pip install opencv-python
pip install pandas
pip install testresources
pip install ipywidgets omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops wandb
pip install matplotlib
pip install timm
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

git clone https://github.com/CompVis/latent-diffusion.git
move latent-diffusion latent_diffusion
type nul > latent_diffusion/__init__.py

git clone https://github.com/CompVis/taming-transformers
pip install -e ./taming-transformers

mkdir content
mkdir content\models

curl.exe --output content/models/256x256_diffusion_uncond.pt --url https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
curl.exe --output content/models/512x512_diffusion_uncond_finetune_008100.pt --url https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt
curl.exe --output content/models/secondary_model_imagenet_2.pth --url https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth
curl.exe --output content/models/slip_base_100ep.pt --url https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt

