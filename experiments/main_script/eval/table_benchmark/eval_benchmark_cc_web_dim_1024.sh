for i in {1..12..1}
do  
    incr_min=$(printf %08d $(($i*10000)) )
    outdir="jobs/table_benchmark_dim_1024"

    echo "$outdir/model/m$incr_min.pth"
    python check_file.py --load_path "$outdir/model/m$incr_min.pth"
    OMP_NUM_THREADS=1 python eval_e2e.py \
        --imple_flag "eval" \
        --load_path "$outdir/model/m$incr_min.pth" \
        --feats_load_dir "features/cc_web_resnet50_l4imac" \
        --gpu 0 \
        --batch_size 1 \
        --base_normalize \
        --pca_whitening \
        --pca_reduction 1024 \
        --suppression "vvs" \
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
        --distractor_sampling_ratio "20_50" \
        --save_path "$outdir" \
        --dataset 'cc_web' \
        --feature_extract 'resnet50_l4imac'
done
