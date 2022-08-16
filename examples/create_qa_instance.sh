python tools/create_pre_train_data.py \
       --train_corpus data_for_qa/ft_data_for_qa.txt \
       --output_file training_data_qa.json \
       --tem_pool_path tem_qa_ft_data/ \
       --metrics_file metrics_qa_ft_data.json \
       --num_thread 8 \
       --inject 0 \
