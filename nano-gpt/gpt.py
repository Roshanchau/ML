import torch;
import torch.nn as nn;
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # in how many batches will be tokens represented in and processed parallely
block_size = 256 # sequence length(how many tokens can be represented in one batch)
max_iterations = 5000 # how many times each batch is processed and the weights are updated
eval_interval = 500 # after how many iterations the model should be evaluated
learning_rate = 3e-4 # step with which the weights should be updated (how big , for eg: 0.003 as updated_weight = weight - learning_rate * gradient(change in loss wrt to change in weight i.e derivative of loss wrt to w))
device = 'cuda' if torch.cuda.is_available() else 'cpu'; # use cuda(gpu) if available else use cpu
eval_iters = 200 # number of iterations during evaluation(update parameters after each evaluation)
n_embd= 384 # dimension of embedding(i.e dimension with which a token is represented)
n_head = 6 # number of attention layers(multi-head attention)

# hence: head_size = n_embd /n_head = 64 (i.e each head has dimension of 64 that should be processed)

dropout = 0.2 # Randomly drop some activations during training to reduce overfitting

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out= {}
    model.eval()
    for split in ['train' , 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X , Y = get_batch(split)
            logits , loss = model(X , Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        # turn (B, T , C) (64 , 256 , 384) ----->(B, T , Hs) (64 , 256 , 64(head_size))
        # use_case of nn.Linear = Take every 384-dimensional token representation and transform it into a 64-dimensional representation.

        self.key = nn.Linear(n_embd , head_size , bias=False)
        self.query = nn.Linear(n_embd , head_size , bias= False)
        self.value = nn.Linear(n_embd , head_size , bias= False)

        self.register_buffer('tril' , torch.tril(torch.ones(block_size , block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self , x):
        # input of size (batch, time-step(block-size) , channels(embedding dim))
        # output of size (batch , time-step , head size)
        B, T , C = x.shape
        k = self.key(x) #(B, T , hs)
        q = self.query(x) #(B, T , hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2 , -1) * k.shape[-1] ** -0.5 # (B, T , hs) @ ( B, hs , T) -> (B, T , T)
        wei = wei.masked_fill(self.tril[:T , :T] == 0 , float('-inf')) #(B, T , T)
        wei = F.softmax(wei, dim= -1) #(B, T , T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) #(B, T , hs)
        out = wei @ v # (B, T , T) @ (B, T , hs) -> (B, T , hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multi heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads , n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # give same input x to each heads , they will calculate individual value
        out = torch.cat([h(x) for h in self.heads] , dim= -1)
        # project all 6 head's 64 dim into 384 dim value
        out = self.dropout(self.proj(out))
        return out;

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self , n_embd):
        super().__init__()
        # each token in a block is passed linearly through each layer of nn and activation and dropout
        self.net = nn.Sequential(
            nn.Linear(n_embd , 4* n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd , n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
