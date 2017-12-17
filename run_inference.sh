export CUDA_VISIBLE_DEVICES=0 # only use 1 gpu

dataA=photo
dataB=monet

AtoB=${dataA}2${dataB}
AtoB_model=pretrained/${AtoB}.pb
BtoA=${dataB}2${dataA}
BtoA_model=pretrained/${BtoA}.pb

mkdir -p results/${AtoB}
mkdir -p results/${BtoA}

AtoB_files=data/${BtoA}/testB/*.jpg
BtoA_files=data/${BtoA}/testA/*.jpg

for input_data in ${AtoB_files}; do
	filename=`echo ${input_data} | cut -d "/" -f 4`
	output_data=results/${AtoB}/${AtoB}_${filename}
	while [ ! -f "${output_data}" ]; do
		echo "${AtoB}_${filename}"
		python inference.py --model ${AtoB_model} --input "${input_data}" --output "${output_data}" --image_size 256
	done
done

for input_data in ${BtoA_files}; do
	filename=`echo ${input_data} | cut -d "/" -f 4`
	output_data=results/${BtoA}/${BtoA}_${filename}
	while [ ! -f "${output_data}" ]; do
		echo "${BtoA}_${filename}"
		python inference.py --model ${BtoA_model} --input "${input_data}" --output "${output_data}" --image_size 256
	done
done
