p1='repo/10/10120830_v3_1_other_0927newbatch_gray'
cp $p1/ssl.py models
python eval.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar #21min
python eval.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --doTrainEval 3

python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/basic_query/query_total_features.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 100 

python eval_crop.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --croptype 24of4_zic --split query_total #7min
#newdir=os.path.join(os.path.dirname(args.ckpt),f'{args.split}_{args.croptype}
#os.path.join(newdir,f'{args.split}_features_{args.croptype}.npy')
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/query_total_24of4_zic/query_total_features_24of4_zic.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 20
#args.query_features+'_'+args.ref_features.split('/')[-1]+f'{args.n_perquery}pq.csv'

ct='half_grid'
python eval_crop.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --croptype $ct --split query_total
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/query_total_$ct/query_total_features_$ct.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 10

ct='2of6_grid'
python eval_crop.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --croptype $ct --split query_total #24min
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/query_total_$ct/query_total_features_$ct.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 5

ct='half_grid'
python eval_crop.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --croptype $ct --split ref
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/basic_query/query_total_features.npy --ref_features $p1/ref_$ct/ref_features_$ct.npy --n_perquery 10

wget https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
p1='repo/dino/dino_vitbase'
python eval.py --config $p1/params.yaml --output_dir $p1
python eval.py --config $p1/params.yaml --output_dir $p1 --doTrainEval 3

python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/basic_query/query_total_features.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 100

python eval_crop.py --config $p1/params.yaml --output_dir $p1 --croptype 24of4_zic --split query_total 
#newdir=os.path.join(os.path.dirname(args.ckpt),f'{args.split}_{args.croptype}
#os.path.join(newdir,f'{args.split}_features_{args.croptype}.npy')
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/query_total_24of4_zic/query_total_features_24of4_zic.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 20
#args.query_features+'_'+args.ref_features.split('/')[-1]+f'{args.n_perquery}pq.csv'

ct='half_grid'
python eval_crop.py --config $p1/params.yaml --output_dir $p1 --croptype $ct --split query_total
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/query_total_$ct/query_total_features_$ct.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 10

ct='2of6_grid'
python eval_crop.py --config $p1/params.yaml --output_dir $p1 --croptype $ct --split query_total 
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/query_total_$ct/query_total_features_$ct.npy --ref_features $p1/basic_query/ref_features.npy --n_perquery 5

ct='half_grid'
python eval_crop.py --config $p1/params.yaml --output_dir $p1 --croptype $ct --split ref
python make_n_perquery_df_gpu.py --config $p1/params.yaml --query_features $p1/basic_query/query_total_features.npy --ref_features $p1/ref_$ct/ref_features_$ct.npy --n_perquery 10
