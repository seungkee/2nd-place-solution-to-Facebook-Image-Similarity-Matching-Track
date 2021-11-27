python eval.py --config pp1_params.yaml --allgray 1 --doTrainEval 1
python make_train_sim.py --config pp1_params.yaml --query_features basic_query/train_features_vit_large_patch16_384.npy --ref_features basic_query/train_features_vit_large_patch16_384.npy --n_perquery 256
cp basic_query/train_features_vit_large_patch16_384.npy_sim_256.npy /storage1
python eval.py --config pp2_params.yaml --allgray 1 --doTrainEval 1
python make_train_sim.py --config pp2_params.yaml --query_features basic_query/train_features_vit_large_patch16_224.npy --ref_features basic_query/train_features_vit_large_patch16_224.npy --n_perquery 256
cp basic_query/train_features_vit_large_patch16_224.npy_sim_256.npy /storage1
cp basic_query/train_features_vit_large_patch16_224.pt preprocess/train_features.pt
