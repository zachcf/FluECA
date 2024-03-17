import torch
import torch.nn as nn
import torch.nn.functional as F
from moco import ModelMoCo
from simsam import SimSiam
def create_model(architecture='CNN', use_attention=False,use_moco=False, input_dim=21, hidden_dim=128, output_dim=10, sequence_length=200):
    if architecture == 'LSTM':
        model = LSTMModel(input_dim, hidden_dim, output_dim, use_attention) if use_moco==False else ModelMoCo(ModelBase = LSTMModel,args = (input_dim, hidden_dim, output_dim, use_attention))
    elif architecture == 'BiLSTM':
        model = BiLSTMModel(sequence_length, input_dim, hidden_dim, output_dim, use_attention) if use_moco==False else ModelMoCo(ModelBase = BiLSTMModel,args = (input_dim, hidden_dim, output_dim, use_attention,use_moco))
    elif architecture == 'CNN':
        model = CNNModel(sequence_length, input_dim, hidden_dim, output_dim, use_attention) if use_moco==False else ModelMoCo(ModelBase = CNNModel,args = (sequence_length, input_dim, hidden_dim, output_dim, use_attention,use_moco))
    else:
        raise ValueError("Unsupported architecture type")
    return model

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)  # [Batch Size, Sequence Length, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [Batch Size, Sequence Length]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # [Batch Size, Hidden Dim]
        return outputs

class SelfAttention_h(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention_h, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example and head
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_attention):
        super(LSTMModel, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = SelfAttention_h(hidden_dim,4) if use_attention else None
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [Batch Size, Sequence Length, Hidden Dim]
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out)
            out = self.fc(attn_out)
        else:
            out = self.fc(lstm_out[:, -1, :])
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_attention, use_moco=False):
        super(BiLSTMModel, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        if self.use_attention:
            self.attention = SelfAttention(hidden_dim*2) if use_attention else None
            # self.atte_fc = nn.Linear(hidden_dim*2*sequence_length, hidden_dim*2)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.use_moco = use_moco
        if use_moco:
            self.moco_fc = nn.Sequential(
                nn.Linear(hidden_dim*2, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128, bias=True)
            )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [Batch Size, Sequence Length, 2*Hidden Dim]

        if self.use_attention:
            features = self.attention(lstm_out)
            # features = features.view(features.shape[0], -1)
            # features = self.atte_fc(features)
        else:
          
            h_n_forward = lstm_out[:, -1, :self.hidden_dim]
            h_n_backward = lstm_out[:, 0, self.hidden_dim:]
            features = torch.cat((h_n_forward, h_n_backward), dim=1)

        out = self.fc(features)

        if self.use_moco:
            features = self.moco_fc(features)
            return out,features
        else:
            return out

class CNNModel(nn.Module):
    def __init__(self, sequence_length, input_dim, hidden_dim, output_dim, use_attention, use_moco=False):
        super(CNNModel, self).__init__()
        self.use_attention = use_attention
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        fc_dim = hidden_dim * (sequence_length // 2)
        self.attention = SelfAttention(fc_dim) if use_attention else None
        self.fc = nn.Linear(fc_dim, output_dim)
        self.use_moco = use_moco
        if use_moco:

            self.moco_fc = nn.Sequential(
                nn.Linear(fc_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128, bias=True)
            )

    def forward(self, x):
        # Swap dimensions for CNN
        x = x.permute(0, 2, 1)
        conv_out = self.pool(F.relu(self.conv1(x)))
        features = conv_out.view(conv_out.size(0), -1)

        if self.use_attention:
            features = features.unsqueeze(1)
            features = self.attention(features)
            # features = features.squeeze(1)
        out = self.fc(features)

        if self.use_moco:
            features = self.moco_fc(features)
            return out,features
        else:
            return out
