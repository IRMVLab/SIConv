python evaluate_scannet.py  \
    --gpu 0 \
    --model pointconv_weight_density_n16 \
    --data_root /data/dataset/scannet_pickle \
    --batch_size 16 \
    --num_point 8192 \
    --model_path train_result/scannet_xyz_2022_08_10_15_32_29/best_model_epoch_1000.ckpt \
    --dump_dir test_result/dump \
    > log_test_SIConv_net.txt 2>&1 &

