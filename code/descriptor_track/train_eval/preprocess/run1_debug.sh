python makeDataFolder.py
python make_imlist_debug_small.py
#python make_imlist.py
python make_fake_public_ground_truth.py
#cp public_ground_truth.csv /facebook/data/public_ground_truth.csv
./download_facebook_debug.sh
python makeClassFolder.py /facebook/data/images/train1M/train
#cp -rf AugLy /facebook2/AugLy
#cp -rf noto-emoji /facebook2/noto-emoji
#git clone https://github.com/facebookresearch/AugLy.git /facebook2/AugLy
#git clone https://github.com/googlefonts/noto-emoji.git /facebook2/noto-emoji
