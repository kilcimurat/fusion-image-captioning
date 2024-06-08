import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualAttention(nn.Module):
    """Some Information about VisualAttention"""
    def __init__(self, units):
        super(VisualAttention, self).__init__()
        self.W1 = nn.Linear(units, units)
        self.W2 = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, x, hidden):
        # x shape == (N, 64, 2048)

        # hidden shape == (1, N, hidden_size)

        # hidden shape == (N, 1, hidden_size)
        hidden = hidden.squeeze().unsqueeze(1)

        # attention hidden layer shape == (N, 64, units)
        attention_hidden_layer = torch.tanh(self.W1(x) + self.W2(hidden))
        # score shape == (N, 64, 1)
        score = self.V(attention_hidden_layer)
        # attention weights shape == (N, 1, 64)
        attention_weights = torch.softmax(score, axis=1).squeeze().unsqueeze(1)
        # context vector shape == (N, 2048)
        context_vector = torch.bmm(attention_weights, x).squeeze().unsqueeze(0)

        return context_vector, attention_weights

class GNMT(nn.Module):
    """GNMT"""
    def __init__(self, feature_size, hidden_size, output_size, dropout_p=0.1):
        super(GNMT, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.linear = nn.Linear(feature_size, hidden_size)

        self.rnn_0 = nn.LSTM(2*self.hidden_size, 2*self.hidden_size)
        self.rnn_1 = nn.LSTM(2*self.hidden_size, 2*self.hidden_size)
        self.rnn_2 = nn.LSTM(2*self.hidden_size, 2*self.hidden_size)
        self.rnn_3 = nn.LSTM(2*self.hidden_size, 2*self.hidden_size)

        self.classifier = nn.Linear(2*self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attention = VisualAttention(self.hidden_size)

    def forward(self, x, hidden_states, carry_states, features):

        x = x.unsqueeze(0)
        x = self.embedding(x)
        x = self.dropout_p(x)
        features = self.linear(features).unsqueeze(0)

        x = torch.cat((x, features), dim=-1)
        x, hidden_0 = self.rnn_0(x, (hidden_states[0], carry_states[0]))
        #context_vector, attention_weigths = self.attention(features, hidden_0[0])

        #x = torch.cat((x, context_vector), 2)
        x = self.dropout_p(x)
        x, hidden_1 = self.rnn_1(x, (hidden_states[1], carry_states[1]))

        residual = x
        x = self.dropout_p(x)
        x, hidden_2 = self.rnn_2(x, (hidden_states[2], carry_states[2]))
        x = x + residual

        residual = x
        x = self.dropout_p(x)
        x, hidden_3 = self.rnn_3(x, (hidden_states[3], carry_states[3]))
        x = x + residual



        x = self.softmax(self.classifier(x[0]))
        hiddens = [hidden_0[0], hidden_1[0], hidden_2[0], hidden_3[0]]
        carrys = [hidden_0[1], hidden_1[1], hidden_2[1], hidden_3[1]]


        return x, hiddens, carrys


