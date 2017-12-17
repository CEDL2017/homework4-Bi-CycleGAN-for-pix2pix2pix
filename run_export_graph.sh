export CUDA_VISIBLE_DEVICES=0 # only use 1 gpu

dataX=photo
dataY=monet
dataZ=ukiyoe

datetime=
checkpoint_dir=checkpoints/${datetime}

python export_graph.py --checkpoint_dir ${checkpoint_dir} --XtoY_model ${dataX}2${dataY}.pb --YtoX_model ${dataY}2${dataX}.pb --XtoZ_model ${dataX}2${dataZ}.pb --ZtoX_model ${dataZ}2${dataX}.pb --image_size 256
