export CUDA_VISIBLE_DEVICES=0 # only use 1 gpu

dataX=$1 #photo
dataY=$2 #monet
dataZ=$3 #ukiyoe

datetime=$4 #20170101-0000
checkpoint_dir=checkpoints/${datetime}

python export_graph.py --checkpoint_dir ${checkpoint_dir} --XtoY_model ${dataX}2${dataY}.pb --YtoX_model ${dataY}2${dataX}.pb --XtoZ_model ${dataX}2${dataZ}.pb --ZtoX_model ${dataZ}2${dataX}.pb --image_size 256
