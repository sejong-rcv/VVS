OMP_NUM_THREADS=1 python train_e2e.py \
    --imple_flag "train" \
    --dataset "vcdb" \
    --feats_load_dir "features/vcdb_resnet50_l4imac" \
    --gpu 0 \
    --batch_size 1 \
    --base_normalize \
    --pca_whitening \
    --pca_reduction 1024 \
    --suppression "vvs" \
    --save_path "jobs/table_benchmark_dim_1024" \
    --learning_rate 2e-5 \
    --feature_extract 'resnet50_l4imac' \
    --vvs_vinit "v_t" \
    --vvs_h_connect \
    --vvs_sigmoid_T 512.0 \
    --refinement \
    --vvs_tsm \
    --vvs_tgm \
    --vvs_ddm \
    --weight_triplet 1.0 \
    --weight_saliency 1.0 \
    --weight_frame 1.0 \
    --distractor_sampling_ratio "20_50" 