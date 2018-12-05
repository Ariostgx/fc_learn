#python joint_tune.py --method 'supervised' --lr 0.01 --gpu 1 --evaluate_freq 5 --epochs 50 --sources 'market' --batch_size 16
python split.py --batch_size 16 --method 'split_pure_3' --num_share_block 3 --gpu 1 --epochs 200 --lr 0.01
