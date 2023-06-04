# base
python train.py --config base_config

# Random Masking Replacement
python train.py --config random_masking_replacement1_config

# Random Masking Insertion
python train.py --config random_masking_insertion1_config

# Adverb Augmentation
python train.py --config adverb_config

# AEDA
python train.py --config aeda_config

# EDA
python train.py --config random_deletion_config
python train.py --config random_insertion_config
python train.py --config random_swap_config
python train.py --config random_synonym_replacement_config