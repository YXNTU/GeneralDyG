import argparse

parser = argparse.ArgumentParser(description='Denoise')
parser.add_argument('--dir_data', type=str, default='./dataset')
parser.add_argument('--name_pos', type=str, default='EU3')
parser.add_argument('--ratio_neg', type=str, default='1')
parser.add_argument('--data_set', type=str, default='btc_alpha',
                    choices=('wikipedia', 'reddit', 'wadi', 'btc_otc', 'btc_alpha'))
parser.add_argument('--neg', type=str, default='01')  # 01 05 1
parser.add_argument('--max_len', type=int, default=24)  # wikipedia 24

##data param
parser.add_argument('--batch_size', type=int, default=128)  # 256
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--num_data_workers', type=int, default=0)  # 25
parser.add_argument('--gpus', type=int, default=1)

##model param
parser.add_argument('--ckpt_file', type=str, default='./')
parser.add_argument('--input_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=258)  # 354
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--drop_out', type=float, default=0.4)
parser.add_argument('--n_layer', type=int, default=6, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--seed', type=int, default=95540)

args = parser.parse_args()
