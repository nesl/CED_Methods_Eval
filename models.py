import math
import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer):
        super().__init__()
        self.lstm_net = nn.LSTM(input_size=input_dim,
                                hidden_size=hidden_dim,
                                num_layers=num_layer,
                                bidirectional=False,
                                batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm_net(x)
#         x = x[:,-1, :] # if label is simple
        x = self.fc(x)
        logits = torch.sigmoid(x)
        return logits



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs.transpose(-1,1))  # input should have dimension (N, C, L)
        o = self.linear(y1.transpose(-1,1)) # input to Linear layer should have dimension (N, L, C)
#         o = self.linear(y1[:, :, -1]) # for simple label
        return torch.sigmoid(o)



class TSTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim:int, num_head=8, num_layers=6, pos_encoding=False, dropout=0.0):
        '''Time-Series Transformer'''
        super().__init__()
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.pos_encoder = PositionalEncoder(d_model=input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src, mask=None):
        if self.pos_encoding:
            x = self.pos_encoder(src)
        x = self.transformer_encoder(src, mask=mask)  # src should have dimension (N, S, E)
        x = self.fc(x)
        logits = torch.sigmoid(x)
        return logits

    
class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i < d_model - 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x


class StateEncoder(torch.nn.Module): # Temporary encoder
    def __init__(self, n_state, out_dim=32):
        super().__init__()  
        self.n_state = n_state
        self.out_dim = out_dim

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.n_state)
        x = x.repeat_interleave(4, dim=-1, output_size=self.out_dim)
        x.to(torch.float32)

        return x 


class MultiTaskTSTransformer(nn.Module):
    def __init__(self, in_dim_embed: int, out_dim_ce: int, out_dim_state: int, hidden_dim_state=32, num_head=8, num_layers=6, pos_encoding=False, dropout=0.0):
        '''
        Time-Series Tansformer with two task heads - one for CE label and one for state prediction
        '''
        super().__init__()

        in_dim = in_dim_embed + hidden_dim_state

        self.state_encoder = StateEncoder(n_state=out_dim_state)
        self.pos_encoding = pos_encoding
        if self.pos_encoding:
            self.pos_encoder = PositionalEncoder(d_model=in_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_0 = nn.Linear(in_dim, out_dim_ce)
        self.fc_1 = nn.Linear(in_dim, out_dim_state)

    def forward(self, src, s_src, mask=None, src_key_padding_mask=None):
        '''
        Arguments:
            src (float32, Tensor): Processed sensor embedding.
            s_src (int64, Tensor): Current states of FSMs.
            mask (bool, Tensor): Causal mask for the self-attention block.
        '''
        s_src = self.state_encoder(s_src)
        src = torch.cat([src, s_src], dim=-1)
        if self.pos_encoding:
            x = self.pos_encoder(src)
        x = self.transformer_encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)  # src should have dimension (N, S, E)
        x1 = self.fc_0(x)
        x2 = self.fc_1(x)
        o1 = torch.sigmoid(x1)
        o2 = torch.sigmoid(x2)

        return o1, o2
