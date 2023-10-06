export CUDA_VISIBLE_DEVICES=3,2
export WANDB_CONSOLE=off
export TORCH_DISTRIBUTED_DEBUG=INFO

python train.py\
    --run_name flexible_score_model\
    --test_sigma_intervals\
    --esm_embeddings_path data/esm2_3billion_embeddings.pt\
    --log_dir workdir\
    --cache_path data/dataset_cache\
    --lr 1e-3\
    --tr_sigma_min 0.1\
    --tr_sigma_max 19\
    --rot_sigma_min 0.03\
    --rot_sigma_max 1.55\
    --ns 48\
    --nv 10\
    --num_conv_layers 6\
    --dynamic_max_cross\
    --scheduler plateau\
    --scale_by_sigma\
    --dropout 0.1\
    --remove_hs\
    --c_alpha_max_neighbors 24\
    --receptor_radius 15\
    --num_dataloader_workers 64\
    --num_workers 64\
    --cudnn_benchmark\
    --num_inference_complexes 500\
    --use_ema\
    --distance_embed_dim 64\
    --cross_distance_embed_dim 64\
    --sigma_embed_dim 64\
    --scheduler_patience 30\
    --n_epochs 850\
    --batch_size 56\
    --all_atoms \
    --wandb \
    # --limit_complexes 64\
    # --no_chi_angle \
    # --val_inference_freq 5\
    # --config workdir/flexible_score_model/model_parameters.yml