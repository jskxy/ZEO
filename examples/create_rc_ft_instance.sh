python tools/create_pre_train_data.py \
       --train_corpus data_for_rel/train_data_for_rel.txt \
       --output_file training_data_rel.json \
       --tem_pool_path tem_rel_ft_data/ \
       --metrics_file metrics_rel_ft_data.json \
       --inject 0 \
       --num_thread 8 \
