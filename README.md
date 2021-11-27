## Installation
```bash
git clone https://github.com/seungkee/2nd-place-solution-to-Facebook-Image-Similarity-Matching-Track.git
sudo docker pull pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
sudo docker run --runtime=nvidia --rm -it --ipc=host --gpus all -v $pwd/2nd-place-solution-to-Facebook-Image-Similarity-Matching-Track:/submission pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
```

## Run Setting & Train & Inference
```bash
cd /submission
chmod +x run.sh
./run.sh
```

## Hardware Requirements
Local Drive Capacity >= 4TB, A100 40GB GPU x 8

