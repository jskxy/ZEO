python tools/create_pre_train_data.py \
       --train_corpus data_for_ner/ner_ft_data.txt \
       --output_file training_data_ner.json \
       --tem_pool_path tem_ner_ft_data/ \
       --metrics_file metrics_ner_ft_data.json \
       --inject 0 \
       --num_thread 8 \
