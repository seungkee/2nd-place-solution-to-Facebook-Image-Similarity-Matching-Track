[Run Setting & Train & Inference]
git clone https://github.com/seungkee/2nd-place-solution-to-Facebook-Image-Similarity-Matching-Track.git
sudo docker pull pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
sudo docker run --runtime=nvidia --rm -it --ipc=host --gpus all -v $pwd/2nd-place-solution-to-Facebook-Image-Similarity-Matching-Track:/submission pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
cd /submission
chmod +x run.sh
./run.sh

[HardWare Requirements]
Local Drive Capacity >= 4TB
GPU : A100 40GB x 8
