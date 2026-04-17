import numpy as np
import torch

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch: {left.shape} != {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
         q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
         gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
         gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
         gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
         q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
         gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
         gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
         gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
         gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
         gpt.trf_blocks[b].att.out_proj.bias = assign(
         gpt.trf_blocks[b].att.out_proj.bias,
         params["blocks"][b]["attn"]["c_proj"]["b"])
         gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
         gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
         gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
         gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
         gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
         gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
         gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
         gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    GPT文本生成函数（支持温度采样、Top-K采样、提前终止）
    :param model: GPT模型
    :param idx: 输入的token索引张量 [batch_size, seq_len]
    :param max_new_tokens: 最多生成多少个token
    :param context_size: 模型最大上下文长度
    :param temperature: 温度系数（>0 采样，=0 贪心搜索）
    :param top_k: Top-K采样，保留概率最高的K个token
    :param eos_id: 结束符token，遇到则停止生成
    :return: 生成完成的完整token序列
    """
    # ===================== 核心修复：设备同步 =====================
    # 自动获取模型所在设备（CPU/GPU）
    device = next(model.parameters()).device
    # 将输入token移动到模型所在设备，解决跨设备报错
    idx = idx.to(device)
    # =============================================================

    # 循环生成新token
    for _ in range(max_new_tokens):
        # 截取最后 context_size 个token作为模型输入（防止超过上下文限制）
        idx_cond = idx[:, -context_size:]

        # 禁用梯度计算（推理阶段不需要更新参数）
        with torch.no_grad():
            logits = model(idx_cond)

        # 只取最后一个位置的输出（预测下一个token）
        logits = logits[:, -1, :]

        # ===================== Top-K 采样 =====================
        if top_k is not None:
            # 获取概率最高的top_k个logit
            top_logits, _ = torch.topk(logits, top_k)
            # 找到top_k中的最小阈值
            min_val = top_logits[:, -1]
            # 把低于阈值的logit设为负无穷（softmax后概率为0）
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),  # 设备同步
                logits
            )

        # ===================== 温度缩放 + 采样 =====================
        if temperature > 0.0:
            # 温度缩放：控制生成多样性
            logits = logits / temperature
            # 转换为概率分布
            probs = torch.softmax(logits, dim=-1)
            # 多项式采样（随机采样，更有创意）
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # 贪心搜索：直接选概率最高的token（最稳定、最保守）
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # ===================== 提前终止（修复bug） =====================
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # 将新生成的token拼接到原序列后面
        idx = torch.cat((idx, idx_next), dim=1)

    # 返回完整的token序列
    return idx