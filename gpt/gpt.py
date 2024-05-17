import torch 
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb


wandb.init(
    project="andrej-llm",
    config={
        "learning-rate":3e-4,
        "epochs" : 100000,
        "batch_size" : 64,
        "context_len" : 256,
        "eval_interval" : 1000,
        "n_embed" : 384,
    }
)


learning_rate = 3e-4
epochs = 100000
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
context_len = 256
eval_interval = 1000
n_layers = 6
n_embed = 384

class LinearForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.linear = nn.Linear(n_embed,4*n_embed)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4*n_embed,n_embed)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,X):
        out = self.proj(self.relu(self.linear(X)))
        out = self.dropout(out)
        return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size, bias=False)
        self.query = nn.Linear(n_embed,head_size, bias=False)
        self.value = nn.Linear(n_embed,head_size, bias=False)
        self.register_buffer("trill",torch.tril(torch.ones(context_len,context_len)))
        self.dropout = nn.Dropout(p=0.2)


    def forward(self,x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (C**0.5)
        wei= wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei= F.softmax(wei, dim=-1)
        wei= self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHead(nn.Module):
    def __init__(self,n_head,head_size):
        super().__init__()
        self.mha = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self,X):
        out = torch.cat([h(X) for h in self.mha],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class TransformerBlock(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        self.mha = MultiHead(n_head, n_embed//n_head)
        self.ff = LinearForward(n_embed)
        self.prenorm = nn.LayerNorm(n_embed)
        self.postnorm = nn.LayerNorm(n_embed)

    def forward(self,X):
        x = self.mha(self.prenorm(X))
        out = self.ff(self.postnorm(x))
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.lookup_table = nn.Embedding(vocab_size,n_embed)
        self.linear_head = nn.Linear(n_embed,vocab_size)
        self.positional_embeddings = nn.Embedding(context_len,n_embed)
        self.transformer = nn.Sequential(*[TransformerBlock(n_embed,n_head=6) for _ in range(n_layers)])
        self.layern = nn.LayerNorm(n_embed,vocab_size)

    def forward(self,idx,targets=None):
        B,T = idx.shape
        token_embed = self.lookup_table(idx)
        pos_embed = self.positional_embeddings(torch.arange(T,device=device))
        x = token_embed + pos_embed
        x = self.transformer(x)
        x = self.layern(x)
        logits = self.linear_head(x)

        B, T, C = logits.shape
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T,C),targets.view(B*T))
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-context_len:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:] # the whole batch, the next token and the whole vocab size
            probs = F.softmax(logits,dim=-1) # this gives us the probabilities
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat([idx, idx_next],dim=1)
        return idx

def get_batch(split):
    ix = torch.randint(len(split) - context_len, (batch_size,))
    x = torch.stack([split[i:i+context_len] for i in ix])
    y = torch.stack([split[i+1:i+context_len+1] for i in ix])
    return x.to(device),y.to(device)

@torch.no_grad()
def evaluate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_interval)
        for epoch in range(eval_interval):
            if split=="train":
                X,y = get_batch(train_data)
            else:
                X,y = get_batch(val_data)
            logits,loss = model(X,y)
            losses[epoch] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

with open("data/input.txt","r",encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set("".join(text))))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}
encode = lambda s:[stoi[c] for c in s]
decode = lambda s:[itos[c] for c in s]
data = torch.tensor(encode(text))
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


model = BigramLanguageModel(vocab_size).to(device)
optimiser = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    # wandb log
    if epoch%eval_interval == 0:
        losses = evaluate_loss()
        print(f"{epoch}:train loss = {losses['train']:.4f} val loss = {losses['val']:.4f}")
    wandb.log({"train_loss":losses['train'],"val_loss":losses['val']})
    
    
    train_Xb,train_yb = get_batch(train_data)
    logits, loss = model(train_Xb,train_yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
    


context = torch.zeros((1,1),dtype=torch.long,device=device)
print("".join(decode(model.generate(context,1000)[0].tolist())))

torch.save(model.state_dict(),"models/gpt.pth")