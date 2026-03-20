import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制（Decoder风格，带因果掩码/下三角掩码）
    核心作用：让模型在处理序列时，能同时关注不同位置的信息，且每个注意力头关注不同维度的特征
    Args:
        d_in: 输入特征维度（每个token的特征数）
        d_out: 输出特征维度（需能被num_heads整除）
        context_length: 最大上下文序列长度（用于生成掩码）
        dropout: Dropout概率，防止过拟合
        num_heads: 注意力头的数量（多头并行计算注意力）
        qkv_bias: Q/K/V线性层是否使用偏置项
    """
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()  # 继承nn.Module的初始化

        # 断言检查：输出维度必须能被注意力头数整除，保证每个头的维度相同
        assert (d_out % num_heads == 0), \
            "d_out必须能被num_heads整除，否则无法均分每个注意力头的维度"

        # 核心参数初始化
        self.d_out = d_out          # 整体输出特征维度
        self.num_heads = num_heads  # 注意力头数量
        self.head_dim = d_out // num_heads  # 单个注意力头的维度

        # Q/K/V投影层：将输入维度d_in映射到输出维度d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Query（查询）投影层
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)    # Key（键）投影层
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)  # Value（值）投影层

        self.out_proj = nn.Linear(d_out, d_out)  # 多头输出拼接后的最终投影层
        self.dropout = nn.Dropout(dropout)       # Dropout层，随机失活部分注意力权重

        # 注册因果掩码（不参与梯度更新的缓冲区）
        # torch.triu生成上三角矩阵，diagonal=1表示对角线以上为1（未来token），对角线及以下为0
        # 作用：屏蔽未来的token，让当前token只能关注自身及之前的token
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        前向传播：实现多头注意力的核心计算逻辑
        Args:
            x: 输入张量，形状为 [batch_size, num_tokens, d_in]
               - batch_size: 批次大小（一次处理的样本数）
               - num_tokens: 当前序列的token数量（≤context_length）
               - d_in: 每个token的输入特征维度
        Returns:
            context_vec: 注意力输出张量，形状为 [batch_size, num_tokens, d_out]
        """
        # 获取输入张量的维度：b=批次大小，num_tokens=当前序列长度，d_in=输入特征维度
        b, num_tokens, d_in = x.shape

        # 第一步：生成Q/K/V矩阵（将输入投影到d_out维度）
        queries = self.W_query(x)  # Q矩阵，形状 [b, num_tokens, d_out]
        keys = self.W_key(x)       # K矩阵，形状 [b, num_tokens, d_out]
        values = self.W_value(x)   # V矩阵，形状 [b, num_tokens, d_out]

        # 第二步：拆分Q/K/V为多个注意力头
        # 维度变化：[b, num_tokens, d_out] → [b, num_tokens, num_heads, head_dim]
        # 把d_out维度拆分为num_heads个head_dim，实现多头并行计算
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 第三步：调整维度顺序，将num_heads提到第二个维度
        # 维度变化：[b, num_tokens, num_heads, head_dim] → [b, num_heads, num_tokens, head_dim]
        # 目的：让每个注意力头独立计算，batch和head维度在前，便于矩阵运算
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # 第四步：计算注意力分数（Q @ K^T）
        # K.transpose(2, 3)：将K的num_tokens和head_dim维度交换，形状变为 [b, num_heads, head_dim, num_tokens]
        # 注意力分数形状：[b, num_heads, num_tokens, num_tokens]
        # 每个位置(i,j)表示第i个token对第j个token的注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 第五步：应用因果掩码，屏蔽未来的token
        # 截取掩码到当前序列长度（mask[:num_tokens, :num_tokens]），转为布尔型
        # mask_bool中为True的位置是未来token，需要被屏蔽
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 将未来token的注意力分数设为负无穷（softmax后权重为0，完全不关注）
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 第六步：计算注意力权重（缩放 + softmax + dropout）
        # 缩放因子：head_dim的平方根，防止注意力分数过大导致softmax饱和
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 对注意力权重应用dropout，随机失活部分权重，增强泛化能力
        attn_weights = self.dropout(attn_weights)

        # 第七步：计算上下文向量（注意力权重 × V矩阵）
        # 维度变化：[b, num_heads, num_tokens, num_tokens] @ [b, num_heads, num_tokens, head_dim]
        # → [b, num_heads, num_tokens, head_dim]
        context_vec = (attn_weights @ values)
        # 调整维度顺序，将num_heads放回原位置：[b, num_heads, num_tokens, head_dim] → [b, num_tokens, num_heads, head_dim]
        context_vec = context_vec.transpose(1, 2)

        # 第八步：拼接所有注意力头的输出
        # contiguous()确保张量内存连续，避免view报错
        # 维度变化：[b, num_tokens, num_heads, head_dim] → [b, num_tokens, d_out]（num_heads*head_dim=d_out）
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 第九步：最终投影层，对拼接后的多头输出做线性变换
        context_vec = self.out_proj(context_vec)

        return context_vec