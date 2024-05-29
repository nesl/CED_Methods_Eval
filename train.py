import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from func_utils import set_seeds, focal_loss, create_src_causal_mask, CEDataset, train, test
from models import RNN, TCN, TimeSeriesTransformer
 
parser = argparse.ArgumentParser(description='NN Model Evaluation')
parser.add_argument('model', type=str, choices=['lstm', 'tcn', 'transformer', 'ae_lstm', 'ae_tcn', 'ae_transformer'])
parser.add_argument('dataset', type=int, help='Dataset size', choices=[2000, 4000, 6000, 8000, 10000])
parser.add_argument('seed',  type=int, help='Random seed') #0, 17, 1243, 3674, 7341, 53, 97, 103, 191, 99719
parser.add_argument('-p', '--pylon', action='store_true', help='use PYLON constraint during training')

args = parser.parse_args()

set_seeds(args.seed)


""" Setting """

batch_size = 256
n_epochs = 2000
learning_rate = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = focal_loss(alpha=torch.tensor([.005, 0.45, 0.45, 0.45]),gamma=2)


""" Load datasets """

if args.model == 'lstm' or args.model == 'tcn' or args.model == 'transformer':
    train_data_file = './CE_dataset/ce5min_train_data_{}.npy'.format(args.dataset)
    train_label_file = './CE_dataset/ce5min_train_labels_{}.npy'.format(args.dataset)
    test_data_file = './CE_dataset/ce5min_test_data.npy'
    test_label_file = './CE_dataset/ce5min_test_labels.npy'

elif args.model == 'ae_lstm' or args.model == 'ae_tcn' or args.model == 'ae_transformer':
    train_data_file = './CE_dataset/ae2ce5min_train_data_{}.npy'.format(args.dataset)
    train_label_file = './CE_dataset/ae2ce5min_train_labels_{}.npy'.format(args.dataset)
    test_data_file = './CE_dataset/ae2ce5min_test_data.npy'
    test_label_file = './CE_dataset/ae2ce5min_test_labels.npy'

else:
    raise Exception("Unknown dataset.")

ce_train_data = np.load(train_data_file)
ce_train_labels = np.load(train_label_file)
ce_test_data = np.load(test_data_file)
ce_test_labels = np.load(test_label_file)

print(train_data_file)
print(ce_train_data.shape, ce_train_labels.shape, ce_test_data.shape, ce_test_labels.shape)

ce_train_dataset = CEDataset(ce_train_data, ce_train_labels)
ce_train_loader = DataLoader(ce_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
ce_test_dataset = CEDataset(ce_test_data, ce_test_labels)
ce_test_loader = DataLoader(ce_test_dataset, batch_size=batch_size, shuffle=False)



""" Load NN models """

input_dim = ce_train_data.shape[-1]
output_dim = 4

if args.model == 'lstm' or args.model == 'ae_lstm':
    model = RNN(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_layer=3)

elif args.model == 'tcn' or args.model == 'ae_tcn':
    model = TCN(input_size=input_dim, output_size=output_dim, num_channels=[256,256,256,256,256], kernel_size=2, dropout=0.2)

elif args.model == 'transformer':
    model = TimeSeriesTransformer(input_dim=input_dim, output_dim=output_dim, num_head=4, num_layers=6, pos_encoding=True)

elif args.model == 'ae_transformer':
    model = TimeSeriesTransformer(input_dim=input_dim, output_dim=output_dim, num_head=1, num_layers=6, pos_encoding=True)

else:
    raise Exception("Model is not defined.") 

summary(model)


""" Training and Testing """

src_causal_mask = create_src_causal_mask(ce_train_data.shape[1]) if args.model == 'transformer' or args.model == 'ae_transformer' else None

train(
    model=model,
    train_loader=ce_train_loader,
    val_loader=ce_test_loader, # need to generate validation data
    n_epochs=n_epochs,
    lr=learning_rate,
    criterion=criterion,
    src_mask=src_causal_mask,
    device=device
    )

torch.save(model.state_dict(), 'saved_models/{}-{}-{}.pt'.format(args.model, args.dataset, args.seed))

test(
    model=model,
    data_loader=ce_test_loader,
    criterion=criterion,
    src_mask=src_causal_mask
    )