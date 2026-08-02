"""Turn user histories into padded next-item training tensors.  PART A.

Key idea: a user's history ``[i1, i2, ..., iL]`` is treated exactly like a
sentence. The model predicts the next item at every position, so the training
input is ``[i1..i_{L-1}]`` and the target is ``[i2..iL]`` (shifted by one). We
right-pad every sequence to ``max_len`` with ``pad_id``.

Why right-padding needs no attention mask: the model is causal, so a real
position only ever attends to *earlier* positions -- and with padding on the
right, everything earlier is real. Pad positions are ignored in the loss
(``ignore_index=pad_id``), so they never affect training.
"""

"""
Transformer 推荐系统数据预处理(Sequential Recommendation)

目标:

把每个用户的历史行为（点击、购买、浏览等）转换成 Transformer 可以训练的数据。

核心思想和 GPT 完全一样:

    GPT:
        输入前面的单词
            ↓
        预测下一个单词（Next Token Prediction）

    推荐系统:
        输入用户之前交互过的商品
            ↓
        预测用户下一件可能交互的商品（Next Item Prediction）

因此:

    一个用户历史  =  一句话(sentence)
    一个商品ID    =  一个token

步骤1:构造 Input 和 Target（整体右移一位）

例如用户历史:
    history = [3, 8, 5, 10]

转换为:
    Input  = [3, 8, 5]
    Target = [8, 5,10]

也就是:
    Input = history[:-1]
    Target = history[1:]

模型学习:
    看到 3        → 预测 8
    看到 3 8      → 预测 5
    看到 3 8 5    → 预测 10

这和 GPT 学习:
    I        → love
    I love   → deep
    I love deep → learning
完全一样。

步骤2:为什么要保留 max_len + 1 个商品?

Transformer 一次最多只能输入 max_len 个 token。
例如:max_len = 4

如果用户历史很长:[1,2,3,4,5,6]不能全部输入。

应该保留最近:max_len + 1

也就是:[2,3,4,5,6]

为什么要 +1?因为之后要整体右移:

Input:
    [2,3,4,5]
Target:
    [3,4,5,6]
这样 Input 和 Target 都刚好长度为 max_len。

步骤3:为什么需要 Padding?不同用户历史长度不同,例如:

User A:[3,8,5]

User B:[7]

User C:[6,1]

Transformer 一次训练要求:所有序列长度必须一致。因此需要 Padding:
    [3,8,5,0,0]
    [7,0,0,0,0]
    [6,1,0,0,0]
其中:
    pad_id = 0表示"这里没有商品,只是补齐长度。"

步骤4:为什么 Padding 放右边?Transformer 使用的是 Causal Attention（因果注意力）。

特点:每个位置只能看到自己左边的内容。例如[3,8,5,0,0],预测商品5时:

模型只能看到: 3,8

不会看到:0,0 因为 Padding 全放在右边。

所以:所有历史位置前面都是真实商品。这种方式最适合 GPT 类模型。

步骤5:为什么不用 Attention Mask?
很多 Transformer 都需要 Attention Mask。但这里不用。

原因:
Padding 全放在最后:[真实商品][真实商品][真实商品][PAD][PAD]
由于 GPT 本身只能看左边,真实商品永远不会看到右边的 PAD。
因此:Causal Attention 已经自动避免了 PAD 的影响。所以:不需要额外 Attention Mask。


步骤6:为什么 Padding 不参与 Loss?
Target 可能是:[8,5,10,0,0]

训练时:Loss 只计算:loss(8)loss(5)loss(10), 最后两个0由于设置ignore_index = pad_id,因此:Padding 不会参与 Loss,也不会更新模型参数。


最终输出是什么?

输入:
{
    user1:[3,8,5,10],
    user2:[7,2],
    user3:[6,1,9]
}

得到:X（模型输入）

[
 [3,8,5,0,0],
 [7,0,0,0,0],
 [6,1,0,0,0]
]

Y（训练目标）

[
 [8,5,10,0,0],
 [2,0,0,0,0],
 [1,9,0,0,0]
]

之后直接送入 Transformer:
    X
      ↓
Transformer
      ↓
预测下一件商品
      ↓
与 Y 比较计算 Loss
      ↓
反向传播训练模型

一句话总结这段代码的作用就是:
把每个用户的历史商品序列,当成 GPT 的一句话,
通过"整体右移一位（Shift）"构造 Input 和 Target,
再统一右侧 Padding,使所有序列长度一致,
最终训练 Transformer 学会:"根据用户过去点击/购买的商品,预测下一件最可能发生的商品。"

本质上:
    Next Token Prediction
            ↓
    换成
    Next Item Prediction
"""

import torch


def build_training_sequences(histories, max_len, pad_id=0):
    """Build right-padded (inputs, targets) tensors from user histories.

    Args:
        histories: {user: [item_id, ...]} chronological, ints in 1..n_items.
        max_len:   sequences are cropped to their last ``max_len + 1`` items
                   (so inputs/targets are at most ``max_len`` long) then padded.
        pad_id:    id used for padding (default 0).

    Returns:
        X: (N, max_len) LongTensor of inputs   (right-padded with pad_id)
        Y: (N, max_len) LongTensor of targets  (right-padded with pad_id)

    One row per user with at least 2 items. Users with <2 items are skipped.
    """
    # 先把每个用户的历史序列转成训练样本；长度小于 2 的用户跳过
    sequences = []

    for history in histories.values():
        if len(history) < 2:
            continue

        # 保留最近的 max_len + 1 个物品，保证后面做 shift 后仍然不超过 max_len
        seq = history[-(max_len + 1):]

        # 输入是去掉最后一个元素的序列，目标是去掉第一个元素的序列
        # 例如 [1, 2, 3, 4] -> X=[1, 2, 3], Y=[2, 3, 4]
        x = seq[:-1]
        y = seq[1:]

        # 如果长度不足 max_len，就在右边补 pad_id；如果太长，就截断到 max_len
        if len(x) < max_len:
            x = x + [pad_id] * (max_len - len(x))
            y = y + [pad_id] * (max_len - len(y))
        else:
            x = x[-max_len:]
            y = y[-max_len:]

        sequences.append((x, y))

    # 如果没有有效样本，返回空张量，形状为 (0, max_len)
    if not sequences:
        return torch.empty(0, max_len, dtype=torch.long), torch.empty(0, max_len, dtype=torch.long)

    X = torch.tensor([x for x, _ in sequences], dtype=torch.long)
    Y = torch.tensor([y for _, y in sequences], dtype=torch.long)
    return X, Y


def get_batch(X, Y, batch_size, generator=None):
    """Sample ``batch_size`` random rows from (X, Y).

    Returns (xb, yb), each (batch_size, max_len). Sampling with replacement is
    fine. Use ``generator`` for reproducibility.
    """
    # 从数据中随机抽取 batch_size 个样本，允许重复抽样
    # 这里用 randint，样本编号会在 [0, N) 之间均匀采样
    if batch_size <= 0:
        return X[:0], Y[:0]

    idx = torch.randint(0, X.shape[0], (batch_size,), generator=generator)
    return X[idx], Y[idx]

