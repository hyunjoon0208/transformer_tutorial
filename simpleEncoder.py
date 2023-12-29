import torch

# 메소드로 만들어도 됨.
def multi_head_attention(self, Q, K, V):
    # Q, K, V: [batch_size, n_heads, seq_len, d_k]
    num_batch, num_head, num_token_length, att_dim = K.shape

    Q /=(att_dim**0.5)
    attention_score = Q@ K.permute(0,1,3,2) # [num_batch, num_head, num_token_length, num_token_length]

    attention_score = torch.softmax(attention_score, dim=3)

    Z = attention_score @ V # [num_batch, num_head, num_token_length, att_dim]

    return Z, attention_score



class MultiHeadAttention(torch.nn.Module):
    def __ini__(self):
        super().__init__()

    def forward(self, Q, K, V):
        # Q, K, V: [batch_size, n_heads, seq_len, d_k]
        num_batch, num_head, num_token_length, att_dim = K.shape

        Q /=(att_dim**0.5)
        attention_score = Q@ K.permute(0,1,3,2) # [num_batch, num_head, num_token_length, num_token_length]

        attention_score = torch.softmax(attention_score, dim=3)

        Z = attention_score @ V # [num_batch, num_head, num_token_length, att_dim]

        return Z, attention_score


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_head, Dropout_p=0.5, Activation=torch.nn.ReLU()):
        super().__init__()

        self.num_head = num_head
        self.hidden_dim = hidden_dim

        assert hidden_dim % num_head == 0

        self.MHA = MultiheadAttention()
        # self.MM = multi_head_attention()

        self.W_Q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)      

        self.W_O = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.LayerNorm1 = torch.nn.layerNorm(self.hidden_dim)
        self.LayerNorm2 = torch.nn.layerNorm(self.hidden_dim)

        self.Dropout = torch.nn.Dropout(p = Dropout_p)

        self.Linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.Linear2 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.Activation = Activation



    def to_multi_head_attention(self, vector):
        num_batch, num_token_length, hidden_dim = vector.shape
        att_dim = hidden_dim // self.num_head
        vector = vector.view(num_batch, num_token_length, self.num_head,att_dim) # [num_batch, num_token_length, num_head, att_dim]
        vector = vector.permute(0,2,1,3) # [num_batch, num_head, num_token_length, att_dim]
        # return vector


    def forward(self, input_Q, input_K, input_V):
        # input_Q, input_K, input_V: [batch_size, seq_len, hidden_dim] ==> self_attention
        # input_Q = [num_batch, num_token_length, hidden_dim]

        Q = self.W_Q(input_Q)
        K = self.W_K(input_K)
        V = self.W_V(input_V)

        #multi head attention

        # hidden_dim = 64
        # num_head = 8
        # att_dim = hidden_dim // num_head # att_dim = hidden_dim / num_head 트릭? 원래 이렇게 많이씀.

        # split
        # num_batch, num_token_length, hidden_dim = Q.shape
        # att_dim = hidden_dim // self.num_head
        # Q = Q.view(num_batch, num_token_length, self.num_head,att_dim) # [num_batch, num_token_length, num_head, att_dim]

        Q = self.to_multi_head_attention(Q) # [num_batch, num_head, num_token_length, att_dim]
        K = self.to_multi_head_attention(K)
        V = self.to_multi_head_attention(V)


        Z, attention_score = self.MHA(Q, K, V) # [num_batch, num_head, num_token_length, att_dim]



        Z = Z.permute(0,2,1,3) # [num_batch, num_token_length, num_head, att_dim]
        # Z = Z.reshape(num_batch, num_token_length, num_head*att_dim) # [num_batch, num_token_length, hidden_dim]
        Z = Z.reshape(num_batch, num_token_length, self.hidden_dim) # [num_batch, num_token_length, hidden_dim]

        Z = self.W_O(Z) 


        # residual connection
        Z = self.LayerNorm1(self.Activation(Z) + input_Q)
        Z1 = self.Dropout(Z)

        # feed forward
        Z = self.Activation(self.Linear1(Z))
        Z = self.Dropout(Z)
        Z = self.Activation(self.Linear2(Z))
        Z = self.Dropout(Z)

        # residual connection
        Z = self.LayerNorm2(Z1 + Z)

        return Z
        