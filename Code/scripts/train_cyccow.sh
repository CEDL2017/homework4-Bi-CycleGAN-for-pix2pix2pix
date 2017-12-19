# python train.py --dataroot ./datasets/cowb2c --name cb2c_cyclegan --model cycle_gan --pool_size 50 --no_dropout
# python train.py --dataroot ./datasets/cowc2r --name cc2r_cyclegan --model cycle_gan --pool_size 50 --no_dropout

python train.py --dataroot ./datasets/cowb2c2r --name cb2c2r_bicyclegan --model bicycle_gan --pool_size 50 --no_dropout --dataset_mode biunaligned
