---
theme: default
title: LLM-Forge
info: |
  A Slidev deck focused on the big picture of LLMs,
  adapted from the uploaded slides and the repository lorendw7/LLM-Forge.
class: text-left
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
mdc: true
---

# LLM-Forge
## Understanding the Big Picture of LLMs

**Repository:** `lorendw7/LLM-Forge`

<div class="text-sm opacity-80 mt-4 leading-relaxed">
Most of the materials in this presentation are based on the GitHub repository <b>LLM-Forge</b> and the book <b>Build a Large Language Model (From Scratch)</b>, with additional explanations adapted from standard LLM / Transformer background knowledge.
</div>

<div class="text-sm opacity-70 mt-3 leading-relaxed">
日本語: 本スライドの大部分は GitHub リポジトリ <b>LLM-Forge</b> と書籍 <b>Build a Large Language Model (From Scratch)</b> に基づき、補足説明として一般的な LLM / Transformer の知識を加えています。
</div>

<div class="mt-6">

- Part I — What an LLM is  
  <span class="opacity-70">LLMとは何か</span>
- Part II — How an LLM learns  
  <span class="opacity-70">LLMはどう学習するか</span>
- Part III — How an LLM becomes useful  
  <span class="opacity-70">LLMはどう実用化されるか</span>

</div>

---
layout: section
---

# 1. The Big Picture
### 全体像

<img src="./images/build_llm.png" class="rounded shadow w-[600px] mx-auto"/>

---

# Why this project matters

`LLM-Forge` is useful because it shows one complete path:

<div class="mt-4">

**text** → **tokens** → **Transformer** → **pretraining** → **fine-tuning**

</div>

Instead of treating LLMs as a black box, it breaks the system into understandable steps.

<div class="mt-6 text-sm opacity-80">
日本語:  
このプロジェクトは、LLMを「魔法」ではなく、段階的に理解できる実装として見せてくれます。
</div>

---

# One sentence summary

## An LLM is a system that:

- converts text into tokens
- predicts the next token from previous tokens
- learns this pattern at scale
- is later adapted for useful tasks

<div class="mt-6 text-sm opacity-80">
日本語:  
LLMとは、過去のトークンから次のトークンを予測し、その能力を大規模学習で獲得し、最後に実用タスクへ適応させるシステムです。
</div>

---

# The 3-stage lifecycle

| Stage | Core question | Outcome |
|---|---|---|
| Build | What is the model architecture? | A GPT-style network |
| Pretrain | How does the model learn language? | A base model |
| Fine-tune | How do we adapt it to tasks? | A useful assistant or classifier |

<div class="mt-5">
This is the main story behind the entire repository.
</div>

---
layout: section
---

# 2. What an LLM is
### LLMとは何か

<img src="./images/decoder.png" class="shadow rounded w-[600px] mx-auto"/>

---

# Core idea: next-token prediction

An autoregressive LLM learns:

$$P(x_t \mid x_{<t})$$

Meaning:

- input: all previous tokens
- output: the next token
- repeated many times across many sequences

<div class="mt-6 text-sm opacity-80">
日本語:  
自己回帰型LLMは、「これまでの並び」を見て「次に来るトークン」を予測します。
</div>

---

# From raw text to training pairs

The dataset pipeline turns text into:

- token IDs
- fixed-length windows
- shifted input / target pairs

```python
input_chunk = token_ids[i:i + max_length]
target_chunk = token_ids[i + 1:i + 1 + max_length]
```

### Intuition
- input = what the model sees
- target = the next-token answers

---

LLMs are pretrained by predicting the next word in a text

<img 
src="./images/data_sliding_window.png"
class="shadow rounded w-[600px] mx-auto"/>

---

# Why tokenization matters

Tokenization is the bridge between language and computation.

Without it:

- the model cannot process text numerically
- training cannot be framed as prediction over a vocabulary

### Big picture
A lot of LLM behavior starts with how text is segmented.

<div class="mt-6 text-sm opacity-80">
日本語: トークン化は、自然言語を計算可能な単位へ変換する入口です。
</div>

---

<img 
src="./images/embedding_pipeline.png"
class="shadow rounded w-[500px] mx-auto"/>

---

# Embeddings: turning IDs into vectors

The model begins with two embeddings:

- **token embedding**: what the token is
- **position embedding**: where the token is

```python
tok_emb = nn.Embedding(vocab_size, emb_dim)
pos_emb = nn.Embedding(context_length, emb_dim)
```

### Why both matter
Meaning alone is not enough; order also matters.

---

# Attention: the central mechanism

Self-attention lets each token look at other tokens and decide what matters.

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Intuition
- Query: what I need
- Key: what I offer
- Value: what I send

---

<img src="./images/attention_mechanisms.png" class="rounded shadow w-[600px] mx-auto"/>

---

# Why causal masking is essential

A GPT-style model must not see future tokens during training.

```python
self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length), diagonal=1)
)
```

### Why it matters
- prevents answer leakage
- matches training with generation
- preserves left-to-right prediction

<div class="mt-6 text-sm opacity-80">
日本語: 因果マスクは、未来情報の漏洩を防ぎ、学習条件と生成条件を一致させます。
</div>

---

<img src="./images/causal_masking.png" class="rounded shadow w-[600px] mx-auto"/>

---

# Why multiple attention heads?

Multi-head attention gives the model multiple views of the same sequence.

Different heads may capture:

- local syntax
- long-range dependency
- separators and structure
- topic continuity

### Big idea
One head is one pattern detector; many heads create richer relational reasoning.

---

<img src="./images/muti_head_attention.png"
 class="rounded shadow w-[600px] mx-auto"/>

---

# A Transformer block in one view

A typical block contains:

1. LayerNorm
2. Multi-head attention
3. Residual connection
4. Feed-forward network
5. Another residual path

### Mental model
- attention = communication across tokens
- FFN = computation inside each token
- residuals = stable information flow

---

<img src="./images/gpt_model.png"
 class="rounded shadow w-[400px] mx-auto"/>

---

# What the model finally outputs

The final layer projects hidden states back to the vocabulary:

```python
self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
```

So at each position, the model produces:

- a score for every token in the vocabulary
- then a probability distribution after softmax

### This is how generation starts.

---

# Part I takeaway

## The architecture story is:

**tokens** → **embeddings** → **attention** → **Transformer blocks** → **logits**

This is the structural heart of a GPT-style LLM.

<div class="mt-6 text-sm opacity-80">
日本語: Part I は、LLMの「構造」を理解する段階です。
</div>

---
layout: section
---

# 3. How an LLM learns
### LLMはどう学習するか

---

# Pretraining in one sentence

Pretraining teaches the model broad language patterns from raw text.

### Why raw text is enough
Because the supervision is built into the sequence itself:

- prefix = input
- next token = target

This is **self-supervised learning**.

---

# The training objective

The standard objective is next-token cross-entropy loss.

### What it does
It pushes the model to assign high probability to the correct next token.

### Why it matters
When repeated over massive data, this simple objective builds surprisingly general language capability.

<div class="mt-6 text-sm opacity-80">
日本語: 単純な次トークン予測でも、大量データで繰り返すと汎用的な言語能力が形成されます。
</div>

---

# The training loop

```python
optimizer.zero_grad()
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
optimizer.step()
```

### Four actions
- clear old gradients
- compute loss
- backpropagate
- update parameters

This is the basic engine of learning.

---

# What makes training stable?

In practice, a working LLM pipeline needs more than a forward pass.

Common ingredients include:

- **AdamW** for optimization
- **learning-rate scheduling** for stable convergence
- **gradient clipping** for guardrails

### Big picture
Architecture gives capacity; optimization turns that capacity into learned behavior.

---

# How do we know the model is improving?

A serious training workflow tracks both:

- training loss
- validation loss
- sampled generations
- tokens seen / compute budget

### Why this matters
Loss curves show learning trends. Samples show actual behavior.

---

# What pretraining really gives us

After pretraining, the model becomes a **base model**.

It usually learns:

- statistical language patterns
- syntax and local structure
- some reusable representations

It does **not automatically** become:

- a reliable assistant
- a task specialist
- a safe product

---

# Part II takeaway

## Pretraining turns structure into capability.

Part I builds the machine.  
Part II teaches the machine language.

<div class="mt-6 text-sm opacity-80">
日本語: Part II は、モデルに言語の規則性を学ばせる段階です。
</div>

---
layout: section
---

# 4. How an LLM becomes useful
### LLMはどう実用化されるか

---

# Why fine-tuning exists

A pretrained model is general, but not yet aligned to a specific use.

Fine-tuning adapts the base model for:

- classification
- instruction following
- domain specialization
- product constraints

### Big picture
Pretraining builds broad knowledge; fine-tuning shapes behavior.

---

# Example 1: classification

In classification, the goal changes.

- pretraining: predict the next token
- classification: output a label

Examples:

- spam vs ham
- sentiment categories
- intent classes

### So what changes?
- data format
- output interpretation
- metrics

---

# Example 2: instruction tuning

Instruction tuning teaches the model to respond in a task-oriented format.

Typical structure:

- instruction
- optional input
- response

### Why it matters
The model learns not only language continuation, but also how to behave like an assistant.

<div class="mt-6 text-sm opacity-80">
日本語: 指示チューニングは、モデルを「続き生成器」から「応答システム」へ近づけます。
</div>

---

# Data format is part of the training signal

Prompt templates are not cosmetic.

They teach the model:

- where the task begins
- what counts as context
- what style the answer should follow

### Important point
In LLM systems, **data formatting** is often as important as model architecture.

---

# Full fine-tuning vs PEFT

There are two broad adaptation strategies:

- **full fine-tuning**: update all parameters
- **PEFT**: update a small subset or added modules

### Why PEFT matters
- lower memory cost
- faster experimentation
- easier to maintain multiple task variants

Examples: LoRA, adapters

---

# From base model to assistant

A useful assistant usually emerges through layers of adaptation:

1. pretraining for general language ability
2. supervised fine-tuning for task behavior
3. sometimes extra alignment and safety stages

### Key idea
A chat model is not only a model architecture. It is a trained and behavior-shaped system.

---

# Part III takeaway

## Fine-tuning turns capability into usefulness.

Part III is where the model starts to match human tasks and product needs.

<div class="mt-6 text-sm opacity-80">
日本語: Part III は、モデルを「使える形」に整える段階です。
</div>

---
layout: section
---

# 5. Final synthesis
### まとめ

---

# The complete LLM story

## A simple way to remember it

- **Build** the architecture
- **Pretrain** on large-scale text
- **Fine-tune** for target behavior

---

# Final takeaway

A compact repo, but a complete story:

**data → attention → GPT → pretraining → fine-tuning**

<div class="mt-6 text-sm opacity-80">
日本語: 小さな実装ですが、LLM開発全体の流れを学べる教材です。
</div>

---

# Sources and Attribution

## Main materials used in this slide deck

- **Primary repository:** `lorendw7/LLM-Forge`
- **Book structure referenced:** *Build a Large Language Model (From Scratch)*
- **Supporting materials:**
  - project README and source files
  - notebooks for pretraining, classification, and instruction tuning
  - standard Transformer / GPT concepts used for explanation

> Most of the materials in this presentation are based on the GitHub repository **LLM-Forge** and the book *Build a Large Language Model (From Scratch)*, with additional explanations adapted from standard LLM / Transformer background knowledge.

<span class="opacity-70">
本スライドの大部分は GitHub リポジトリ **LLM-Forge** と書籍 *Build a Large Language Model (From Scratch)* に基づき、補足説明として一般的な LLM / Transformer の知識を加えています。
</span>

---

# Thank You

## LLM-Forge as a learning path

A compact repo, but a complete story:

**data → attention → GPT → pretraining → fine-tuning**
