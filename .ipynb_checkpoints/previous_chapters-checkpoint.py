import torch
import torch.nn as nn
import tiktoken 

from torch.utils.data import Dataset, DataLoader
        

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout, context_len, qkv_cache=False):
        super().__init__()
        self.heads = heads
        self.head_dim = emb_dim//heads
        
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), 1))

        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        B, T, C = x.shape
        print("In multi head attention: ", x.shape)

        query = self.Wq(x).reshape(B, T, self.heads, self.head_dim)
        query = query.transpose(1,2) # B, H, T, D
        
        
        keys = self.Wk(x).reshape(B, T, self.heads, self.head_dim)
        keys = keys.transpose(1,2) # B, H, T, D
        
        values = self.Wv(x).reshape(B, T, self.heads, self.head_dim)
        values = values.transpose(1,2) # B, H, T, D


        attn_scores = query @ keys.transpose(-1,-2) # B, H, T, T
        mask = self.mask[:T, :T].bool()
        
        print("attn_scores shape: ", attn_scores.shape)
        print("mask shape ", mask.shape)
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights= torch.softmax(attn_scores/values.shape[-1]**0.5, dim=-1)
        
        attn_weights = self.dropout(attn_weights)

        context_vect = (attn_weights @ values).transpose(1,2).reshape(B, T, C)

        return self.out_proj(context_vect)

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, dropout_rate, context_len):
        super().__init__()
        self.ffn = FeedForwardN(emb_dim)
        self.mha = MultiHeadAttention(emb_dim, heads, dropout_rate, context_len, False)
        self.drop_embeds = nn.Dropout(dropout_rate)
        self.norm1 = LayerNorm(emb_dim) # need to define this
        self.norm2 = LayerNorm(emb_dim) #need to define this


    def forward(self, x):
        shortcut = x  
        x = self.norm1(x)
        x = self.mha(x)
        x = self.drop_embeds(x)
        x = x+shortcut 

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        print("FFn out: ", x.shape)
        x = self.drop_embeds(x)
        x = x + shortcut 

        return x


class FeedForwardN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.l1 = nn.Linear(emb_dim,  emb_dim*4)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(emb_dim*4, emb_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        var = torch.var(x, correction=0, keepdims=True)

        x_norm = (x-mean)/var+self.eps
        return (x_norm*self.scale) + self.shift
        
class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_len"], config["emb_dim"])
        self.drop_embeds = nn.Dropout(config["dropout_rate"])
        
        self.traf_blocks = nn.Sequential(*[TransformerBlock(config["emb_dim"], config["heads"], config["dropout_rate"], config["context_len"]) for _ in range(config["tranf_blocks"])])
        
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"])

    def forward(self, Ints):
        bs, seq_len = Ints.shape
        tok_embeds = self.tok_emb(Ints)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=Ints.device))

        x = tok_embeds + pos_embeds 
        x = self.drop_embeds(x)
        print("Going to Traf Block:", x.shape)
        x = self.traf_blocks(x)

        x= self.final_norm(x)
        return self.out_head(x)   


def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        print("idx shape: ",idx.shape, "context_len: ", context_size)
        idx_cond = idx[:, -context_size:]
        print("idx_cond shape: ", idx_cond.shape)

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        probabs = torch.softmax(logits, dim=-1)

        idx_next = torch.argmax(probabs, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx
    

class RMSNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.shift = nn.Parameters(torch.ones(dim))
        self.eps = 1e-5

    def forward(self, x):
        rms = torch.mean(x**2, dim=-1) + self.eps 
        rms_root = torch.sqrt(rms)
        return x/rms_root * self.shift
        

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) >= max_len, "total len of tokens must be grater than context len"

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunks = token_ids[i:i+max_len]
            target_chunks = token_ids[i+1:i+max_len+1]

            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(txt, bs=4, max_len=256, stride=128, 
                      shuffle=True, drop_last=True, 
                      num_workers=0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDataset(txt, tokenizer, max_len, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    ) 

    return dataloader
