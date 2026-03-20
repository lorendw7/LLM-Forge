# 导入PyTorch的神经网络模块
import torch
import torch.nn as nn

# 从自定义模块导入多头注意力机制类
from MultiHeadAttention import MultiHeadAttention


class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Units) 激活函数
    这是Transformer/LLM中最常用的激活函数之一，相比ReLU更平滑，效果更好
    实现的是原始论文中的近似计算公式（而非精确的高斯累积分布）
    """

    def __init__(self):
        super().__init__()  # 初始化父类nn.Module

    def forward(self, x):
        """
        GELU的前向计算（近似公式）
        Args:
            x: 输入张量，任意形状
        Returns:
            激活后的输出张量，形状与输入一致
        """
        return 0.5 * x * (1 + torch.tanh(
            # 计算根号(2/π)，作为缩放系数
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            # GELU近似公式的核心部分：x + 0.044715 * x³
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Transformer中的前馈神经网络（FFN）层
    经典结构：线性层（升维）→ GELU激活 → 线性层（降维）
    是Transformer Block的核心组成部分之一
    """

    def __init__(self, cfg):
        """
        初始化前馈网络
        Args:
            cfg: 配置字典，需包含 "emb_dim"（嵌入维度）
        """
        super().__init__()  # 初始化父类nn.Module

        # 构建前馈网络的层序列
        self.layers = nn.Sequential(
            # 第一层线性层：将输入维度从emb_dim升维到4*emb_dim（行业标准缩放系数）
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # GELU激活函数：引入非线性变换
            GELU(),
            # 第二层线性层：将维度从4*emb_dim降维回emb_dim，保持输入输出维度一致
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, emb_dim]
        Returns:
            输出张量，形状与输入一致 [batch_size, seq_len, emb_dim]
        """
        return self.layers(x)

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