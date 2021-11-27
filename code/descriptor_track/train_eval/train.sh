p2='repo/10/10120830_v3_1_other_0927newbatch_gray'
p1='repo/10/10120830_v3_1_other_gray'
cp $p1/ssl.py models
python train.py --config $p1/params.yaml --repo_path $p1 --iters 15500
cp $p2/ssl.py models
python train.py --config $p2/params.yaml --repo_path $p2 --ckpt $p1/checkpoint.pth.tar --iters 12500
