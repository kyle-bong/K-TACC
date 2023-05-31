# base
python train.py --config base_config

# RMR
python train.py --config random_masking_replacement1_config

python train.py --config random_masking_replacement2_config

# RMI
python train.py --config random_masking_insertion1_config
python train.py --config random_masking_insertion2_config

# adverb
python train.py --config adverb_config

# adea
python train.py --config aeda_config

# EDA
python train.py --config random_deletion_config
python train.py --config random_insertion_config
python train.py --config random_swap_config
python train.py --config random_synonym_replacement_config
