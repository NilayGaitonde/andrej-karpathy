import torch 
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt



learning_rate = 1e-4
epochs = 100000
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
context_len = 10
eval_interval = 1000
print(torch.backends.mps.is_available())
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.lookup_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits = self.lookup_table(idx)
        B, T, C = logits.shape
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(B*T,C),targets.view(B*T))
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
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

with open("../data/input.txt","r",encoding='utf-8') as f:
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
    if epoch%eval_interval == 0:
        losses = evaluate_loss()
        print(f"{epoch}:train loss = {losses['train']:.4f} val loss = {losses['val']:.4f}")
    
    train_Xb,train_yb = get_batch(train_data)
    logits, loss = model(train_Xb,train_yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()


context = torch.zeros((1,1),dtype=torch.long,device=device)
print("".join(decode(model.generate(context,1000)[0].tolist())))