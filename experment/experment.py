# python main.py --brats --deep_supervision --depth 6 --filter 64 96 128 192 256 384 512 --min_fmap 2 --batch_size 3 --scheduler --learning_rate 0.0003 --epochs 600 --fold 0 --amp --gpus 1 --task 13 --save_ckpt

# python main.py --gpus 1 --amp --save_preds --exec_mode predict --brats --deep_supervision --depth 6 --filter 64 96 128 192 256 384 512 --min_fmap 2 --tta

#python main.py --brats --batch_size 2 --scheduler --learning_rate 0.0001 --epochs 1000 --fold 2 --amp --gpus 2 --task 13 --save_ckpt --tb_logs --ckpt_name v5

# python main.py --gpus 1 --amp --save_preds --exec_mode predict --brats --tta
