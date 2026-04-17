import numpy as np
import torch
from llmArchitecture import generate_text_simple
# 导入matplotlib的pyplot模块，用于绘图，这是Python最常用的可视化库之一
import matplotlib.pyplot as plt
# 从matplotlib的ticker模块导入MaxNLocator，作用是强制x轴刻度显示为整数
# 避免训练轮次(Epochs)出现小数刻度，让图表更易读
from matplotlib.ticker import MaxNLocator

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


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()



def generate_and_print_sample(model, tokenizer, device, start_context):
    # 将模型切换为评估模式：禁用Dropout等训练专属层，保证生成结果稳定
    model.eval()

    # 从模型的位置嵌入层获取上下文窗口大小（即模型支持的最大输入token数）
    context_size = model.pos_emb.weight.shape[0]

    # 将起始文本编码为token ID张量，并移动到指定计算设备（CPU/GPU）
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # 禁用梯度计算：文本生成不需要反向传播，关闭梯度可节省显存、提升速度
    with torch.no_grad():
        # 调用文本生成函数，基于起始上下文生成最多50个新token
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )

    # 将生成的token ID张量解码回人类可读的文本
    decoded_text = token_ids_to_text(token_ids, tokenizer)

    # 打印生成的文本，将换行符替换为空格，实现紧凑的单行打印格式
    print(decoded_text.replace("\n", " "))

    # 生成完成后，将模型切回训练模式，不影响后续训练流程
    model.train()

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 将模型切换为评估模式：禁用Dropout、BatchNorm等训练专属层，保证评估结果稳定可复现
    model.eval()
    # 禁用梯度计算：评估阶段不需要反向传播，关闭梯度可大幅减少内存占用、提升计算速度
    with torch.no_grad():
        # 计算训练集上的平均损失（仅计算eval_iter个批次，避免全量计算耗时过长）
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        ).cpu().item()
        # 计算验证集上的平均损失（同样仅计算eval_iter个批次）
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        ).cpu().item()
    # 评估完成后，将模型切回训练模式，恢复Dropout等层，不影响后续训练
    model.train()
    # 返回训练集损失和验证集损失，用于监控模型过拟合情况
    return train_loss, val_loss

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 初始化列表，用于记录训练损失、验证损失和已处理的token数量
    train_losses, val_losses, track_tokens_seen = [], [], []
    # 初始化已处理token总数和全局训练步数
    tokens_seen, global_step = 0, -1

    # 主训练循环，遍历所有训练轮次（epoch）
    for epoch in range(num_epochs):
        # 将模型设置为训练模式（启用dropout、batch norm等训练专属层）
        model.train()
        # 遍历训练数据集中的每一个批次
        for input_batch, target_batch in train_loader:
            # 清空优化器中累积的梯度，避免上一批次的梯度影响当前批次
            optimizer.zero_grad()
            # 计算当前批次的损失值
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # 反向传播，计算损失对模型参数的梯度
            loss.backward()
            # 根据梯度更新模型的可训练参数（权重）
            optimizer.step()
            # 累加当前批次处理的token总数（input_batch.numel()返回张量元素总数）
            tokens_seen += input_batch.numel()
            # 全局训练步数+1
            global_step += 1

            # 每经过eval_freq步，执行一次模型评估（可选的评估步骤）
            if global_step % eval_freq == 0:
                # 调用评估函数，计算训练集和验证集的损失
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 将训练损失、验证损失、已处理token数记录到对应列表
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印当前训练进度：轮次、步数、训练损失、验证损失
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )

        # 每完成一个epoch，生成并打印一段示例文本，直观观察模型生成效果
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    # 训练结束后，返回记录的训练损失、验证损失和已处理token数
    return train_losses, val_losses, track_tokens_seen