# LLM-Forge

## Project Introduction

This project strictly implements a complete large language model development pipeline following the three-stage framework in the book *Build a Large Language Model (From Scratch)*:

1. Stage 1: Build the LLM Core
   - Implement data sampling and preprocessing pipeline
   - Implement multi-head attention and Transformer decoder-only architecture
   - Complete model initialization and forward propagation

2. Stage 2: Pre-train the Foundation Model
   - Implement autoregressive language model training loop
   - Integrate AdamW optimizer, learning rate scheduler, and gradient clipping
   - Pre-train on unlabeled text corpus to obtain a usable base model

3. Stage 3: Downstream Task Fine-tuning
   - Parameter-efficient fine-tuning based on pre-trained weights
   - Adapt to text classification and build multi-class classifiers
   - Adapt to instruction following and build chat / assistant models


## 项目简介

本项目严格按照《Build a Large Language Model (From Scratch)》一书的三阶段框架，实现了一个完整的LLM开发流程：

1.  **Stage 1: 构建LLM核心**
    - 实现数据采样与预处理流水线
    - 实现多头注意力机制与Transformer decoder-only架构
    - 完成模型初始化与前向传播逻辑

2.  **Stage 2: 预训练基础模型**
    - 实现自回归语言模型训练循环
    - 集成AdamW优化器、学习率调度与梯度裁剪
    - 在无标注文本语料上完成预训练，得到可用于下游任务的基础模型

3.  **Stage 3: 下游任务微调**
    - 基于预训练模型进行参数高效微调
    - 适配文本分类任务，构建多类别分类器
    - 适配指令跟随任务，实现个人助手/对话模型
