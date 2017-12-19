dataA=$1 #ukiyoe
dataB=$2 #photo

X_input_dir=data/${dataA}2${dataB}/trainB
Y_input_dir=data/${dataA}2${dataB}/trainA
X_output_file=data/tfrecords/${dataB}.tfrecords
Y_output_file=data/tfrecords/${dataA}.tfrecords

python build_data.py --X_input_dir ${X_input_dir} --Y_input_dir ${Y_input_dir} --X_output_file ${X_output_file} --Y_output_file ${Y_output_file}
