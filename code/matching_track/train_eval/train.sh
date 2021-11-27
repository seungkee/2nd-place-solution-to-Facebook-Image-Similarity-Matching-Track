p1='repo/0/matching-1007-from-1002'
p2='repo/0/matching-10072221'
p3='repo/0/matching-1009-from-1008-from-1001-nochange'
cp $p1/ssl.py models
python train.py --config $p1/params.yaml --repo_path $p1 --iters 50500
cp $p2/ssl.py models
python train.py --config $p2/params.yaml --repo_path $p2 --ckpt $p1/checkpoint.pth.tar --iters 18500
cp $p3/ssl.py models
python train.py --config $p3/params.yaml --repo_path $p3 --ckpt $p2/checkpoint.pth.tar --iters 30000

p1='repo/1/matching-1007-from-1002-color'
p2='repo/1/matching-10072221-color'
p3='repo/1/matching-1009-from-1008-from-1001-nochange'
cp $p1/ssl.py models
python train.py --config $p1/params.yaml --repo_path $p1 --iters 44500
cp $p2/ssl.py models
python train.py --config $p2/params.yaml --repo_path $p2 --ckpt $p1/checkpoint.pth.tar --iters 29500
cp $p3/ssl.py models
python train.py --config $p3/params.yaml --repo_path $p3 --ckpt $p2/checkpoint.pth.tar --iters 7000

p1='repo/2/matching-0914_000050'
p2='repo/2/matching-0916_000011'
p3='repo/2/matching-0925-2_000008'
p4='repo/2/matching-0929_000002'
p5='repo/2/matching-0930_000002'
p6='repo/2/matching-0930_000003'
p7='repo/2/matching-1001_000002'
p8='repo/2/matching-1008-from-1001-nochange_000002'
p9='repo/2/matching-1009-from-1008-from-1001-nochange_000002'

cp $p1/ssl.py models
python train.py --config $p1/params.yaml --repo_path $p1 --iters 67500
cp $p2/ssl.py models
python train.py --config $p2/params.yaml --repo_path $p2 --ckpt $p1/checkpoint.pth.tar --iters 32000
cp $p3/ssl.py models
python train.py --config $p3/params.yaml --repo_path $p3 --ckpt $p2/checkpoint.pth.tar --iters 15500
cp $p4/ssl.py models
python train.py --config $p4/params.yaml --repo_path $p4 --ckpt $p3/checkpoint.pth.tar --iters 44500
cp $p5/ssl.py models
python train.py --config $p5/params.yaml --repo_path $p5 --ckpt $p4/checkpoint.pth.tar --iters 10500
cp $p6/ssl.py models
python train.py --config $p6/params.yaml --repo_path $p6 --ckpt $p5/checkpoint.pth.tar --iters 46000
cp $p7/ssl.py models
python train.py --config $p7/params.yaml --repo_path $p7 --ckpt $p6/checkpoint.pth.tar --iters 14000
cp $p8/ssl.py models
python train.py --config $p8/params.yaml --repo_path $p8 --ckpt $p7/checkpoint.pth.tar --iters 6000
cp $p9/ssl.py models
python train.py --config $p9/params.yaml --repo_path $p9 --ckpt $p8/checkpoint.pth.tar --iters 9500

