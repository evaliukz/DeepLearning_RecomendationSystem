"""Training loop for the sequential recommender.  PART B.

This is the Project-1 training loop with two differences: the "vocabulary" is
the item set, and padded target positions are ignored in the loss via
``ignore_index=pad_id``. The model is your ``TinyTransformerLM`` used unchanged.
"""

"""
把每个用户的历史行为（点击、购买、浏览等）转换成 Transformer 可以训练的数据。
核心思想和 GPT 完全一样：
    GPT：
        输入前面的单词
            ↓
        预测下一个单词（Next Token Prediction）

    推荐系统：
        输入用户之前交互过的商品
            ↓
        预测用户下一件可能交互的商品（Next Item Prediction）

因此：
    一个用户历史  =  一句话(sentence)
    一个商品ID    =  一个token
"""

import torch
import torch.nn as nn

from .dataset import get_batch


def train_seqrec(model, X, Y, *, steps=800, batch_size=64, lr=3e-3, pad_id=0,
                 seed=1234, log_every=100, log=True):
    """Train ``model`` on padded (X, Y) next-item tensors.

    Given set-up: a seeded generator, Adam, and a CrossEntropyLoss that ignores
    ``pad_id`` targets. For each step:
        1. xb, yb = get_batch(X, Y, batch_size, g)
        2. zero grads
        3. logits = model(xb)                       -> (B, T, vocab)
        4. loss = CE(logits.reshape(-1, vocab), yb.reshape(-1))   (pad ignored)
        5. backward, step
        6. record loss.item()

    Returns ``{"losses": [...], "final_loss": float}``.
    """
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.train()
    losses = []
    # 训练总步数为 steps，每一步都从数据中随机取一个 batch
    for step in range(steps):
        # 1) 采样一个 batch 的输入 X 和目标 Y
        xb, yb = get_batch(X, Y, batch_size, g)
        # 2) 清零梯度，避免上一步的梯度累积
        optimizer.zero_grad()
        # 3) 让模型根据输入序列预测下一个物品的概率分布
        #    输出形状是 (batch_size, seq_len, vocab_size)
        logits = model(xb)
        # 4) 把时间维度和 batch 维度展平成一维，然后和目标序列比较
        #    这里使用 ignore_index=pad_id，这样 padding 的位置不会参与损失计算
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        # 5) 反向传播，更新参数
        loss.backward()
        optimizer.step()
        # 6) 记录当前这一步的损失值，方便后续观察训练曲线
        losses.append(loss.item())
        # 7) 如果开启了日志打印，则每隔 log_every 步输出一次损失
        if log and (step + 1) % log_every == 0:
            print(f"step {step + 1}/{steps}: loss={loss.item():.4f}")

    if not losses:
        return {"losses": losses, "final_loss": 0.0}
    return {"losses": losses, "final_loss": float(losses[-1])}
