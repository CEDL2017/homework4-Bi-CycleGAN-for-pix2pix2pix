export CUDA_VISIBLE_DEVICES=0 # only use 1 gpu

batch_size=1

X=$1 #photo
Y=$2 #monet
Z=$3 #ukiyoe
fileX=data/tfrecords/${X}.tfrecords
fileY=data/tfrecords/${Y}.tfrecords
fileZ=data/tfrecords/${Z}.tfrecords

python train.py --batch_size=${batch_size} --X=${fileX} --Y=${fileY} --Z=${fileZ}
