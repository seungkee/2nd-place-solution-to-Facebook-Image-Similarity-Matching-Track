d_p1='../../descriptor_track/train_eval/repo/10/10120830_v3_1_other_0927newbatch_gray'
csv1=$d_p1'/basic_query/query_total_features.npy_ref_features.npy100pq.csv'
ct='24of4_zic'
csv2=$d_p1'/query_total_'$ct'/query_total_features_'$ct'.npy_ref_features.npy20pq.csv'
ct='half_grid'
csv3=$d_p1'/query_total_'$ct'/query_total_features_'$ct'.npy_ref_features.npy10pq.csv'
ct='2of6_grid'
csv4=$d_p1'/query_total_'$ct'/query_total_features_'$ct'.npy_ref_features.npy5pq.csv'
ct='half_grid'
csv5=$d_p1'/basic_query/query_total_features.npy_ref_features_'$ct'.npy10pq.csv'

d_p1='../../descriptor_track/train_eval/repo/dino/dino_vitbase'
csv6=$d_p1'/basic_query/query_total_features.npy_ref_features.npy100pq.csv'
ct='24of4_zic'
csv7=$d_p1'/query_total_'$ct'/query_total_features_'$ct'.npy_ref_features.npy20pq.csv'
ct='half_grid'
csv8=$d_p1'/query_total_'$ct'/query_total_features_'$ct'.npy_ref_features.npy10pq.csv'
ct='2of6_grid'
csv9=$d_p1'/query_total_'$ct'/query_total_features_'$ct'.npy_ref_features.npy5pq.csv'
ct='half_grid'
csv10=$d_p1'/basic_query/query_total_features.npy_ref_features_'$ct'.npy10pq.csv'

python merge.py $csv1 $csv2 $csv3 $csv4 $csv5 $csv6 $csv7 $csv8 $csv9 $csv10
python merge_half.py $csv3 $csv4 $csv8 $csv9

p1='repo/0/matching-1009-from-1008-from-1001-nochange'
cp basic_merge.csv $p1/basic_merge.csv
cp half_merge.csv $p1/half_merge.csv
python eval.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/basic_merge.csv
python eval_halfcsv.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/half_merge.csv

p1='repo/1/matching-1009-from-1008-from-1001-nochange'
cp basic_merge.csv $p1/basic_merge.csv
cp half_merge.csv $p1/half_merge.csv
python eval.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/basic_merge.csv
python eval_halfcsv.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/half_merge.csv
#python eval_halfcsv.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $csv4 --gridtype 2of6
p1='repo/2/matching-1009-from-1008-from-1001-nochange_000002'
cp basic_merge.csv $p1/basic_merge.csv
cp half_merge.csv $p1/half_merge.csv
python eval.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/basic_merge.csv
python eval_halfcsv.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/half_merge.csv

python merge_score.py

p1='repo/0/matching-1009-from-1008-from-1001-nochange'
cp final_cand_n.csv $p1/final_cand_n.csv
python eval_half.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/final_cand_n.csv

p1='repo/1/matching-1009-from-1008-from-1001-nochange'
cp final_cand_n.csv $p1/final_cand_n.csv
python eval_half.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/final_cand_n.csv

p1='repo/2/matching-1009-from-1008-from-1001-nochange_000002'
cp final_cand_n.csv $p1/final_cand_n.csv
python eval_half.py --config $p1/params.yaml --ckpt $p1/checkpoint.pth.tar --total_df $p1/final_cand_n.csv

python merge_score2.py
