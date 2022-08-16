python tools/create_pre_train_data.py \
       --train_corpus data_for_sum/ft_data_for_sum.txt \
       --output_file training_data_sum.json \
       --tem_pool_path tem_sum_ft_data/ \
       --metrics_file metrics_sum_ft_data.json \
       --num_thread 8 \
       --inject 0 \
