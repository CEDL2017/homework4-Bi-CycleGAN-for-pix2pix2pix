export CUDA_VISIBLE_DEVICES=0 # only use 1 gpu

batch_size=1

X=photo
Y=monet
Z=ukiyoe
fileX=data/tfrecords/${X}.tfrecords
fileY=data/tfrecords/${Y}.tfrecords
fileZ=data/tfrecords/${Z}.tfrecords

python train.py --batch_size=${batch_size} --X=${fileX} --Y=${fileY} --Z=${fileZ}
