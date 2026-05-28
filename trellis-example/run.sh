export mode=LARGE
python generate_trellis_configs.py $mode y
python generate_tables.py $mode
python runner.py --configs $mode
