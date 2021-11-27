apt-get update 
apt-get install -y vim
apt-get install -y git
apt-get install -y wget
cd /submission/code/pytorch-image-models
pip install .
cd /submission/code
pip install --ignore-installed -r requirements.txt
conda install -y -c conda-forge faiss-gpu
apt-get install -y curl
#curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
apt-get install -y unzip
#unzip awscliv2.zip
./aws/install
pip install torchvision==0.10.0 --no-deps
