NAME="name"

CMD="python experiments/bridgedata_offline_gc.py \
    --config experiments/configs/train_config.py:gc_bc \
    --bridgedata_config experiments/configs/data_config.py:all \
    --name $NAME"

$CMD