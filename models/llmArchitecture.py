# 导入PyTorch的神经网络模块
import torch
import torch.nn as nn
import tiktoken
# 从自定义模块导入多头注意力机制类
from MultiHeadAttention import MultiHeadAttention


tokenizer = tiktoken.get_encoding("gpt2")
# %%
# A layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
# %% [markdown]
# #### An implementation of the GELU activation function
# %%
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
# %% [markdown]
# #### A feed forward neural network module
# %%
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    def forward(self, x):
        return self.layers(x)
# %% [markdown]
# #### The transformer block component of GPT
# %%
class TransformerBlock(nn.Module):
    """
    Transformer的核心模块（编码器/解码器块）
    包含：多头自注意力层 + 前馈神经网络层 + 残差连接 + 层归一化
    采用Pre-LN结构（归一化在注意力/前馈层之前）
    """
    def __init__(self, cfg):
        super().__init__()  # 初始化父类nn.Module

        # 1. 初始化多头自注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],          # 输入特征维度（嵌入维度）
            d_out=cfg["emb_dim"],         # 输出特征维度（与输入一致）
            context_length=cfg["context_length"],  # 上下文长度（序列最大长度）
            num_heads=cfg["n_heads"],     # 注意力头数
            dropout=cfg["drop_rate"],     # Dropout概率
            qkv_bias=cfg["qkv_bias"]      # QKV线性层是否使用偏置
        )

        # 2. 初始化前馈神经网络层
        self.ff = FeedForward(cfg)

        # 3. 初始化两个层归一化层（分别用于注意力和前馈层）
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # 4. 初始化残差连接的Dropout层（防止过拟合）
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        前向传播函数（Pre-LN结构）
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, emb_dim]
        Returns:
            输出张量，形状与输入一致
        """
        # ========== 第一部分：多头自注意力层 + 残差连接 ==========
        shortcut = x  # 保存残差连接的原始输入
        x = self.norm1(x)  # 先做层归一化（Pre-LN结构核心）
        x = self.att(x)    # 多头自注意力计算
        x = self.drop_shortcut(x)  # 对注意力输出做Dropout
        x = x + shortcut   # 残差连接（将原始输入与注意力输出相加）

        # ========== 第二部分：前馈神经网络层 + 残差连接 ==========
        shortcut = x  # 保存注意力层输出作为残差连接的输入
        x = self.norm2(x)  # 再次做层归一化
        x = self.ff(x)     # 前馈神经网络计算
        x = self.drop_shortcut(x)  # 对前馈输出做Dropout
        x = x + shortcut   # 残差连接（将注意力输出与前馈输出相加）

        return x  # 返回Transformer块的最终输出

# %% [markdown]
# ### Coding the GPT model
# %%
class GPTModel(nn.Module):
    """
    GPT（Generative Pre-trained Transformer）核心模型实现
    自回归语言模型的基础架构，包含：嵌入层 → 位置编码 → Transformer块堆叠 → 输出头
    核心特点：仅用Transformer解码器结构（自注意力），无编码器，纯自回归生成
    """
    def __init__(self, cfg):
        super().__init__()  # 初始化父类nn.Module

        # 1. 词嵌入层：将token索引映射为固定维度的向量
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 2. 位置嵌入层：将位置索引映射为固定维度的向量（GPT用绝对位置编码）
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 3. 嵌入层Dropout：防止嵌入层过拟合
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 4. 堆叠多个TransformerBlock（核心编码层）
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 5. 最终层归一化：稳定输出分布，提升训练效果
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # 6. 输出头：将嵌入维度映射回词汇表大小，生成每个token的预测概率
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        """
        GPT模型前向传播
        Args:
            in_idx: 输入token索引张量，形状 [batch_size, seq_len]
                    例如：[[12, 34, 56], [78, 90, 11]] （批次大小2，序列长度3）
        Returns:
            logits: 每个token的预测得分，形状 [batch_size, seq_len, vocab_size]
                    logits经过softmax后可得到每个位置的token概率分布
        """
        # 获取输入的批次大小和序列长度
        batch_size, seq_len = in_idx.shape

        # 步骤1：词嵌入 → [batch_size, seq_len, emb_dim]
        tok_embeds = self.tok_emb(in_idx)

        # 步骤2：位置嵌入
        # 生成0到seq_len-1的位置索引 → [seq_len]
        pos_idx = torch.arange(seq_len, device=tok_embeds.device)
        # 位置嵌入 → [seq_len, emb_dim]
        pos_embeds = self.pos_emb(pos_idx)
        # 词嵌入 + 位置嵌入（广播机制：pos_embeds自动扩展为[batch_size, seq_len, emb_dim]）
        x = tok_embeds + pos_embeds

        # 步骤3：嵌入层Dropout
        x = self.drop_emb(x)

        # 步骤4：通过堆叠的TransformerBlock进行特征编码
        x = self.trf_blocks(x)

        # 步骤5：最终层归一化
        x = self.final_norm(x)

        # 步骤6：输出头生成logits
        logits = self.out_head(x)

        return logits

def generate_text_simple(model, idx,
                 max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx