#python efficient_reader/main.py -sample full_train
python efficient_reader/main.py -sample full_train -model full_train_l2reg_5e-05 -forward_only True
python efficient_reader/main.py -sample full_train -model full_train -forward_only True
python efficient_reader/main.py -sample full_train -model full_train_l2reg_0.0005 -forward_only True



# python efficient_reader/main.py -sample full_train
# python efficient_reader/main.py -sample word_distance_pass
# python efficient_reader/main.py -sample word_distance_fail
# python efficient_reader/main.py -sample frequency_pass
# python efficient_reader/main.py -sample frequency_fail
