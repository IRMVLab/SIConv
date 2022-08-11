python train_scannet_IoU.py  \
    --gpu 0 \
    --model pointconv_weight_density_n16 \
    --data_root /data/dataset/scannet_pickle \
    --log_dir train_result/scannet_xyz_ \
    --num_point 8192 \
    --max_epoch 3000 \
    --batch_size 16 \
    --learning_rate 0.001 \
    > log_train_SIConv_net.txt 2>&1 &
    
