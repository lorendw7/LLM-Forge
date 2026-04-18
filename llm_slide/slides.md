---
theme: default
title: LLM-Forge
info: |
  A Slidev deck for the repository lorendw7/LLM-Forge,
  organized into three major parts following the book
  Build a Large Language Model (From Scratch).
class: text-left
highlighter: shiki
lineNumbers: true
drawings:
  persist: false
transition: slide-left
mdc: true
---

# LLM-Forge

## Rebuilding a GPT-style LLM from Scratch

**Repository:** `lorendw7/LLM-Forge`  

Most of the materials in this presentation are based on the GitHub repository **LLM-Forge** and the book **Build a Large Language Model (From Scratch)**, with additional explanations adapted from standard LLM / Transformer background knowledge.

<div>

- Part I — Build the LLM Core  
  <span class="opacity-70">LLMコアの構築</span>
- Part II — Pre-train the Foundation Model  
  <span class="opacity-70">基盤モデルの事前学習</span>
- Part III — Fine-tune for Downstream Tasks  
  <span class="opacity-70">下流タスク向け微調整</span>

</div>

---
layout: section
---

# 0. Why this project matters
### なぜこのプロジェクトが重要か

---

# Project Positioning

`LLM-Forge` is a learning-oriented repository that follows the same **three-stage pipeline** described in **Build a Large Language Model (From Scratch)**:

1. Build the LLM core
2. Pre-train a foundation model
3. Fine-tune for downstream tasks

### Why it is useful

- It turns LLM theory into readable PyTorch code.
- It shows the path from **tokens** to **attention**, then to **training**, and finally to **task adaptation**.
- It is small enough to study, but complete enough to represent a real LLM workflow.

> Japanese note:  
> このプロジェクトは、LLMの理論を「読めるコード」として理解するための実装例です。

---

# Repository-to-Book Mapping

| Book Stage | Main topics in the book | LLM-Forge files |
|---|---|---|
| Part I | Text data, attention, GPT architecture | `GPTDatasetV1.py`, `dataloader.py`, `MultiHeadAttention.py`, `TransformerBlock.py`, `llmArchitecture.py` |
| Part II | Pretraining on unlabeled data | `pretraining.py`, `pretraining.ipynb`, `data/the-verdict.txt` |
| Part III | Classification finetuning + instruction finetuning | `FineTuningForClassification.ipynb`, `Instruction fine-tuning.ipynb`, `instruction-data.json` |

### Key message

The repo is not just a model definition. It covers the **entire lifecycle**:
**data → model → loss → optimization → adaptation**.

---
layout: section
---

# Part I
# Build the LLM Core
### LLMコアの構築

---

# Part I Overview

This part corresponds to the book's early chapters:

- Understanding what an LLM predicts
- Turning raw text into token sequences
- Implementing **causal self-attention**
- Building a **decoder-only Transformer**
- Generating text autoregressively

---

### Core question

How do we transform plain text into a trainable GPT-style network?

> 日本語メモ:  
> 生の文章を、学習可能な GPT 型ネットワークへどう変換するのか？

---

# Knowledge Point 1: Language Modeling Objective

An autoregressive LLM learns:

$$P(x_t \mid x_{<t})$$

Meaning:
- predict the **next token**
- given all previous tokens
- under a **causal constraint**

### Practical interpretation

If the sequence is:

`[I, love, neural, ...]`

then the model uses the first three tokens to predict the next one.

---

### Why this matters

This single objective powers:
- text generation
- completion
- instruction following
- many downstream transfer behaviors

---

# Knowledge Point 2: Tokenization and Sliding Windows

In `GPTDatasetV1.py`, the text is:

1. tokenized into integer IDs
2. split into overlapping windows
3. turned into `(input, target)` pairs

```python
input_chunk = token_ids[i:i + max_length]
target_chunk = token_ids[i + 1:i + 1 + max_length]
```

---

### Intuition

- `input_chunk`: what the model sees
- `target_chunk`: the same sequence, shifted by one token

### Japanese note

- 入力列 = モデルが読むトークン列
- 目標列 = 1トークン先にずらした教師ラベル

---

# Why Stride Matters

The dataset uses a **stride** when creating windows.

### If stride is small
- more overlap
- more training examples
- higher compute cost

### If stride is large
- less overlap
- fewer examples
- faster but potentially less signal

---

### Conceptual trade-off

You are balancing:
- data reuse
- training efficiency
- local context coverage

This is one of the first places where **data engineering affects model quality**.

---

# Knowledge Point 3: Embeddings

Inside `llmArchitecture.py`, GPT begins with:

- **token embedding**: maps token IDs to vectors
- **position embedding**: injects order information

```python
tok_emb = nn.Embedding(vocab_size, emb_dim)
pos_emb = nn.Embedding(context_length, emb_dim)
```

---

### Why both are necessary

Without token embeddings:
- the model cannot represent semantic identity

Without position embeddings:
- the model cannot distinguish word order

### Japanese note

埋め込みは「単語の意味」と「位置情報」をベクトル空間に載せる仕組みです。

---

# Knowledge Point 4: Self-Attention

Self-attention computes how strongly each token should attend to other tokens.

For token representations $X$:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

and attention weights are based on:

$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

### Interpretation

- **Query**: what this token is looking for
- **Key**: what each token offers
- **Value**: the information to aggregate

---

# Knowledge Point 5: Causal Masking

`MultiHeadAttention.py` registers a triangular mask so tokens cannot see the future.

### Why masking is essential

If token `t` could see token `t+1`, training would leak the answer.

### Result

The model learns generation in the same direction it will use at inference time.

```python
self.register_buffer(
    "mask",
    torch.triu(torch.ones(context_length, context_length), diagonal=1)
)
```

> 日本語: 因果マスクにより、未来トークンの情報漏洩を防ぎます。

---

# Knowledge Point 6: Multi-Head Attention

Why multiple heads?

Because different heads can focus on different patterns:

- local syntax
- long-distance dependency
- punctuation or separators
- topic continuity

---

### In the implementation

The output dimension is split across heads:

```python
self.head_dim = d_out // num_heads
```

Then tensors are reshaped into:

- batch
- heads
- tokens
- head dimension

---

### Key idea
Multi-head attention is **representation factorization**.
It gives the model multiple subspaces for relational reasoning.

---

# Knowledge Point 7: Transformer Block

A Transformer block in this repo contains:

1. LayerNorm
2. Multi-head attention
3. Residual connection
4. LayerNorm
5. Feed-forward network
6. Residual connection

### Why residual paths matter

Residuals:
- stabilize deep training
- preserve information flow
- reduce optimization difficulty

---

### Why LayerNorm matters

LayerNorm improves numerical stability and training smoothness.

> Japanese note:  
> 残差接続は深いネットワークでも情報と勾配を通しやすくします。

---

# Knowledge Point 8: Feed-Forward Network and GELU

After attention, each token is passed through a position-wise MLP.

Typical pattern in this repo:

```python
nn.Linear(emb_dim, 4 * emb_dim)
GELU()
nn.Linear(4 * emb_dim, emb_dim)
```

---

### Why this layer exists

Attention mixes information **across tokens**.
The feed-forward network transforms information **within each token representation**.

### Mental model

- attention = communication
- FFN = private computation

---

# Knowledge Point 9: Decoder-Only GPT Architecture

The repo builds a **decoder-only Transformer**, not an encoder-decoder model.

### Why decoder-only?

Because autoregressive generation only needs:
- masked self-attention
- stacked transformer blocks
- a vocabulary projection head

### Architectural flow

`token ids`
→ `token + position embeddings`
→ `N transformer blocks`
→ `final norm`
→ `linear head`
→ `logits`

This is the basic GPT recipe.

---

# Knowledge Point 10: Logits and Next-Token Prediction

The final linear layer maps hidden states back to vocabulary space:

```python
self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)
```

### Output meaning

For every token position, the model produces a score for **every vocabulary item**.

### Then we apply
- softmax for probabilities
- argmax or sampling for generation

### Important distinction

- **training** uses logits for loss computation
- **generation** uses logits for token selection

---

# Knowledge Point 11: Greedy Generation

The helper `generate_text_simple` performs iterative decoding:

1. crop to the valid context window
2. run a forward pass
3. take the last-token logits
4. pick the next token
5. append it and repeat

---

### Why this is enough for learning

Even a simple greedy loop reveals the full inference mechanism of GPT.

### But in real systems

We often add:
- temperature
- top-k / top-p sampling
- repetition penalties
- KV cache

---

# Part I Summary

By the end of Part I, the repo has already implemented the conceptual heart of a GPT model:

- tokenization and supervised next-token pairs
- embeddings
- causal multi-head attention
- transformer blocks
- decoder-only architecture
- autoregressive generation

### Japanese wrap-up

Part I は「LLM がどう動くか」の骨格を作る段階です。

---
layout: section
---

# Part II
# Pre-train the Foundation Model
### 基盤モデルの事前学習

---

# Part II Overview

Now that the architecture exists, the next task is to teach it language patterns from raw text.

This part focuses on:

- loss definition
- batching and dataloading
- optimization
- evaluation during training
- sample generation for qualitative inspection

### Core question

How does a randomly initialized GPT become a useful base model?

---

# Knowledge Point 12: Pretraining Data

The repo includes `data/the-verdict.txt` as an unlabeled corpus.

### Why unlabeled data is enough

For causal language modeling, we do not need manual labels.
The text itself contains the supervision:

- input = prefix
- target = next token

### Big idea

Pretraining converts abundant raw text into a general-purpose language prior.

> 日本語:  
> 事前学習では、人手ラベル不要の「自己教師あり学習」が成立します。

---

# Knowledge Point 13: Cross-Entropy Loss

The standard training objective is next-token cross-entropy.

### What it measures

How far the model's predicted token distribution is from the true next token.

### Why it is appropriate

Because language modeling is a multi-class classification problem over the vocabulary at every position.

### Optimization target

Lower loss means:
- better token prediction
- better sequence modeling
- usually better text generation quality

---

# Knowledge Point 14: Training Loop Mechanics

A standard PyTorch loop in `pretraining.py` looks like this conceptually:

```python
optimizer.zero_grad()
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()
optimizer.step()
```

### Meaning of each step

- `zero_grad()` resets accumulated gradients
- `backward()` computes parameter gradients
- `step()` updates weights

### Japanese note

- 勾配初期化
- 逆伝播
- パラメータ更新

This is the engine of neural learning.

---

# Knowledge Point 15: AdamW Optimizer

The README says the project integrates **AdamW**.

### Why AdamW is widely used for Transformers

- adaptive per-parameter learning rates
- good stability in large parameter spaces
- decoupled weight decay for better regularization

### Conceptual takeaway

Optimizer choice is not just an implementation detail.
It strongly affects:
- convergence speed
- stability
- generalization

---

# Knowledge Point 16: Learning Rate Scheduling

The project also mentions a learning rate scheduler.

### Why scheduling matters

A fixed learning rate can be too aggressive early or too weak later.

Schedulers help by:
- warming up the model
- reducing step size over time
- improving convergence stability

### In LLM training

Learning rate is one of the most sensitive hyperparameters.

> 日本語: 学習率スケジューリングは、学習の安定化と収束改善に直結します。

---

# Knowledge Point 17: Gradient Clipping

The repo states that gradient clipping is part of pretraining.

### Why clip gradients?

Transformers can sometimes produce unstable large gradients.
Gradient clipping limits gradient magnitude to prevent training explosions.

### Benefits

- fewer numerical instabilities
- smoother training
- better control in deeper models

### Simple intuition

Clipping is like placing guardrails on optimization.

---

# Knowledge Point 18: Train vs Validation Loss

A serious training script tracks both:

- **training loss**
- **validation loss**

### Why both are necessary

If training loss goes down but validation loss stalls or rises:
- overfitting may be happening

If both go down:
- the model is learning useful patterns

### Practical lesson

A good LLM workflow is not only about training harder.
It is about **measuring correctly**.

---

# Knowledge Point 19: Tokens Seen as a Scaling Metric

The training code tracks **tokens seen**.

### Why this is important

In LLM work, progress is often measured not only by epochs, but by:

- steps
- tokens processed
- compute budget

### Why tokens are a better unit

Because sequence lengths and batch shapes can vary.
Token count gives a more universal measure of training exposure.

> 日本語: LLM の学習進捗は、epoch より「何トークン見たか」で語られることが多いです。

---

# Knowledge Point 20: Qualitative Evaluation by Sampling

The repo generates sample text after training progress checkpoints.

### Why sample generations matter

Loss is necessary, but it is not the whole story.
Sampling reveals:
- coherence
- repetition issues
- fluency
- mode collapse symptoms

### Two types of evaluation

- **quantitative**: loss curves
- **qualitative**: generated examples

Good practice requires both.

---

# Knowledge Point 21: What Pretraining Really Learns

A pretrained base model learns broad internal structure such as:

- token co-occurrence statistics
- syntax patterns
- local phrase completions
- some world structure reflected in text

### But it does not automatically become

- a reliable classifier
- a safe assistant
- a well-aligned instruction follower

That is why Part III exists.

---

# Part II Summary

Part II turns the architecture from a static network into a language model that has absorbed statistical regularities from text.

### Main concepts mastered

- self-supervised objective
- cross-entropy optimization
- AdamW + scheduler + clipping
- validation tracking
- token-budget thinking
- generation-based inspection

### Japanese wrap-up

Part II は、モデルに「言語の癖」を覚えさせる工程です。

---
layout: section
---

# Part III
# Fine-tune for Downstream Tasks
### 下流タスク向け微調整

---

# Part III Overview

After pretraining, the model becomes a **base model**.
To make it task-useful, we adapt it.

In this repo, two directions are emphasized:

1. **Text classification**
2. **Instruction following**

### Core question

How do we turn general language knowledge into task-specific capability?

---

# Knowledge Point 22: Transfer Learning

Fine-tuning works because pretraining has already learned useful representations.

### Transfer learning idea

Instead of learning from scratch for every task:
- reuse pretrained weights
- adapt them with task-specific data

### Why this is powerful

It reduces:
- data requirements
- training cost
- convergence time

### Japanese note

転移学習では、事前学習済み表現を再利用して効率よく適応します。

---

# Knowledge Point 23: Fine-tuning for Classification

The notebook `FineTuningForClassification.ipynb` uses the **SMS Spam Collection** dataset.

### Task definition

Input: text message  
Output: label such as `ham` or `spam`

### Key conceptual shift

Pretraining predicts the next token.  
Classification predicts a **task label**.

### What must change?

- dataset format
- prediction head or output interpretation
- loss function target semantics
- evaluation metrics

---

# Knowledge Point 24: Classification Data Pipeline

The notebook downloads and prepares labeled data.

### Why this matters

Unlike pretraining, downstream finetuning depends on:
- explicit labels
- train/validation/test splits
- task-specific batching rules

### General pipeline

raw examples  
→ cleaned dataset  
→ tokenized inputs  
→ model-ready tensors  
→ classification loss

This is where NLP engineering meets supervised learning.

---

# Knowledge Point 25: Classification Head Design

There are several common ways to do GPT-based classification:

- use the last token representation
- use a special token representation
- pool hidden states
- add a linear classification layer

### Key insight

The base GPT is a sequence model.  
For classification, we need to compress sequence information into a label decision.

### Japanese note

系列表現を最終的に「クラス判定」へ写像する設計が分類ヘッドです。

---

# Knowledge Point 26: Metrics Beyond Loss

For classification, loss alone is insufficient.
We also care about metrics such as:

- accuracy
- precision
- recall
- F1 score

### Why accuracy appears in this repo context

The repository includes an `accuracy-plot.pdf`, indicating classification performance tracking.

### Lesson

Downstream tasks need **task-appropriate evaluation**, not only generic LM loss.

---

# Knowledge Point 27: Instruction Fine-tuning

The notebook `Instruction fine-tuning.ipynb` formats data into an instruction-response structure.

Typical prompt template style:

- instruction
- optional input
- desired response

### What this teaches the model

Not just to continue text, but to:
- follow a task framing
- respect user intent
- produce answer-shaped outputs

> 日本語: 指示チューニングは「続き予測」を「指示応答」へ変換する工程です。

---

# Knowledge Point 28: Prompt Formatting as Supervision

Instruction tuning is not only about model weights.  
It is also about **data format**.

### Why formatting matters

The model learns structure from repeated patterns such as:

- `### Instruction:`
- `### Input:`
- `### Response:`

### Important insight

Prompt templates are part of the training signal.
They teach the model what roles different text segments play.

---

# Knowledge Point 29: Supervised Fine-Tuning (SFT)

Instruction tuning here is a form of **supervised fine-tuning**.

### SFT objective

Given an instruction-formatted input, maximize the likelihood of the correct response.

### Why SFT matters

It is usually the first alignment step after pretraining.
It makes a base model more useful in assistant-style interactions.

### But SFT is not the final step in modern systems

Real production chat models may also add:
- preference optimization
- rejection sampling
- safety tuning
- reward modeling

---

# Knowledge Point 30: Parameter-Efficient Fine-Tuning

The project README mentions **parameter-efficient fine-tuning**.

### General idea

Instead of updating every model parameter:
- update only a subset
- or add small trainable modules

### Why this matters in practice

It reduces:
- memory usage
- compute cost
- storage for multiple task variants

---

### Typical examples in the ecosystem

- LoRA
- adapters
- partial-layer tuning

> 日本語: 少ない追加パラメータで適応するのが PEFT の発想です。

---

# Knowledge Point 31: From Base Model to Assistant

A pretrained model knows language patterns.  
An instruction-tuned model behaves more like an assistant.

### Capability shift

Base model:
- good continuation engine
- weak task framing

Instruction-tuned model:
- better compliance with requests
- clearer answer boundaries
- more useful conversational behavior

---

### Important nuance

This is not “new intelligence.”  
It is **behavioral shaping** through targeted supervision.

---

# Knowledge Point 32: End-to-End LLM Lifecycle

The biggest educational value of `LLM-Forge` is that it connects three normally separated views:

- model architecture
- optimization pipeline
- downstream adaptation

### This is the real lifecycle

**Part I** builds the machine  
**Part II** teaches the machine language  
**Part III** teaches the machine how to be useful

### Japanese wrap-up

- Part I: 機械を作る
- Part II: 言語を覚えさせる
- Part III: 役に立つ形へ適応させる

---

# Final Takeaways

## What this project teaches especially well

- How GPT-style LLMs are built from first principles
- Why attention, masking, and residual design matter
- How pretraining differs from finetuning
- Why data format is as important as architecture
- How one codebase can express the full LLM pipeline

## Best reading strategy

1. Start with dataset construction
2. Trace attention tensor shapes carefully
3. Follow the forward pass of `GPTModel`
4. Then study the pretraining loop
5. Finally compare classification vs instruction tuning

---

# Suggested Discussion Questions

1. Why is causal masking indispensable for autoregressive training?
2. What does multi-head attention gain over a single-head design?
3. Why is pretraining possible without manual labels?
4. What changes when moving from next-token prediction to classification?
5. Why does prompt formatting matter during instruction tuning?
6. When should we prefer parameter-efficient finetuning over full finetuning?

---
layout: section
---

# Discussion Guide
### Suggested answers and technical explanations
### 討論用の参考解答と技術解説

---

# Discussion Guide 1
## Causal masking and multi-head attention

### Q1. Why is causal masking indispensable?

**English answer**
Causal masking is indispensable because an autoregressive model must predict token *t* using only tokens before *t*. If future tokens are visible, the model learns from leaked answers and the training setup no longer matches generation time.

**Technical explanation**
- an upper-triangular mask blocks attention to future positions
- each position can attend only to tokens at positions `<= t`
- this preserves the left-to-right factorization of language modeling

**日本語**
因果マスキングは、自己回帰モデルが各時刻 *t* で**過去トークンのみ**を使って予測するために不可欠です。未来トークンが見えると情報漏洩が起き、学習時と推論時の条件が一致しなくなります。

---

### Q2. What does multi-head attention gain over a single-head design?

**English answer**
Multi-head attention lets the model represent several kinds of token relationships at the same time. One head may focus on local syntax, while another may focus on long-range dependency, separators, or topic continuity.

**Technical explanation**
- each head has separate projection matrices for `Q`, `K`, and `V`
- attention maps are learned in parallel in different subspaces
- the concatenated outputs increase expressive power and robustness

**日本語**
マルチヘッド注意の利点は、複数種類の依存関係を**並列に**捉えられることです。あるヘッドは局所文法、別のヘッドは長距離依存や話題継続を捉えることができます。

---

# Discussion Guide 2
## Self-supervision and task transition

### Q3. Why is pretraining possible without manual labels?

**English answer**
Pretraining is possible because language modeling is self-supervised. The corpus already provides the label: if the input is a token prefix, the next token is automatically the training target.

**Technical explanation**
- input: `x_0, x_1, ..., x_{t-1}`
- target: `x_t`
- no human annotation is required because raw text defines the supervision signal itself

**日本語**
事前学習が手作業ラベルなしで成立するのは、言語モデリングが**自己教師あり学習**だからです。入力が接頭列なら、次トークンが自動的に正解ラベルになります。

---

### Q4. What changes when moving from next-token prediction to classification?

**English answer**
The major change is the task objective. Pretraining predicts the next token at every position, while classification compresses the whole sequence into a label such as `spam` or `ham`.

**Technical explanation**
- data changes from unlabeled text to labeled examples
- output changes from vocabulary logits to class logits
- evaluation shifts from language-model loss to metrics such as accuracy, precision, recall, and F1

**日本語**
次トークン予測から分類に移ると、最も大きく変わるのは**目的関数**です。事前学習は各位置で次トークンを予測しますが、分類では系列全体から最終ラベルを出力します。

---

# Discussion Guide 3
## Instruction tuning and PEFT

### Q5. Why does prompt formatting matter during instruction tuning?

**English answer**
Prompt formatting matters because it becomes part of the supervision. Repeated structures such as `Instruction`, `Input`, and `Response` teach the model where the task begins, where extra context appears, and what part should be generated as the answer.

**Technical explanation**
- instruction tuning is still next-token training over structured text
- delimiters and section headers act like role markers
- cleaner templates usually improve consistency and controllability

**日本語**
指示チューニングでプロンプト形式が重要なのは、それ自体が**学習信号の一部**になるからです。`Instruction`、`Input`、`Response` のような構造が、役割境界をモデルに教えます。

---

### Q6. When should we prefer parameter-efficient finetuning over full finetuning?

**English answer**
We prefer PEFT when GPU memory, compute budget, storage, or deployment flexibility is limited. It is especially useful when we want several task-specific variants without storing a full copy of the model each time.

**Technical explanation**
- full finetuning updates all parameters and is expensive for large models
- PEFT methods such as LoRA update only a small set of trainable parameters
- this lowers cost while keeping adaptation practical for many tasks

**日本語**
GPUメモリ、計算予算、保存容量、デプロイ柔軟性が限られている場合は、フル微調整より **PEFT** が適しています。LoRA のような手法は少数の追加パラメータのみを更新するため、低コストで複数タスクに適応できます。

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

**data → attention → GPT → pretraining → finetuning**

<span class="opacity-70">小さな実装で、LLM開発全体像を学べる教材です。</span>
