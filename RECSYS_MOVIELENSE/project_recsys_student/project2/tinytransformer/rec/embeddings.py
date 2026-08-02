"""Extract learned item embeddings from a trained model.  PART C.

Your model actually holds **two** vectors per item:

  * the *input* embedding  ``model.tok_emb.weight``  -- item as context
  * the *output* embedding ``model.head.weight``     -- item as prediction target
    (row ``i`` of the final linear layer is the vector whose dot product with a
    hidden state produces item ``i``'s logit)

Both are learned only from co-occurrence in user histories -- genre is never a
training signal. But they do not behave the same for similarity: items that are
*predicted in the same situations* end up with similar OUTPUT rows, so the
output embedding is what clusters cleanly by genre. Comparing the two is one of
the analysis questions -- that is why ``which`` is a parameter.
"""
"""
    前面的训练阶段让模型学习：用户看到哪些商品以后，下一步通常会点击或购买什么商品。训练完成后，每个商品都会对应一个向量。这个向量可以理解成模型眼里这个商品的“行为特征”。
    这些特征不是人工写进去的，也没有直接告诉模型商品类型，而是从用户历史中的共现和顺序关系自动学出来的。
    从训练好的推荐模型中提取每个物品的 Embedding，方便后续做相似度计算、聚类、PCA 或可视化。

    模型中每个物品有两套向量。为什么每个商品有两套 Embedding？

    1. model.tok_emb.weight：
       Input Embedding，表示物品作为用户历史上下文时的向量。

    2. model.head.weight：
       Output Embedding，表示物品作为“下一件待预测物品”时的向量。
       如果两个物品经常在相似的用户历史后出现，它们的 Output Embedding
       通常会更接近，因此默认使用 output 做物品相似度分析。

    which="output" 时提取 model.head.weight；
    which="input" 时提取 model.tok_emb.weight。

    Vocabulary 的第 0 行对应 pad_id=0，不是真实物品，因此使用 weight[1:]
    去掉 Padding 行。去掉后，Embedding 第 0 行对应 item_id=1，
    第 1 行对应 item_id=2，以此类推。

    detach() 表示提取出的向量只用于分析，不再参与反向传播；
    cpu() 表示将向量移动到 CPU，方便交给 NumPy、sklearn 或 matplotlib 使用。

    如果 normalize=True，则对每个物品向量进行 L2 归一化，使每一行长度为 1。
    这样后续用矩阵乘法计算点积时，结果就等价于 Cosine Similarity，
    避免向量长度影响相似度比较。
    """


#这个函数从模型里取出一整张商品 Embedding Matrix
def item_embeddings(model, which="output", normalize=False):
    """Return an (n_items, d_model) item-embedding matrix, dropping pad row 0.

    Args:
        model: a trained TinyTransformerLM (vocab_size == n_items + 1).
        which: ``"output"`` -> ``model.head.weight`` (default, best for
               similarity), or ``"input"`` -> ``model.tok_emb.weight``.
        normalize: if True, L2-normalize each row.

    Returns:
        emb: (n_items, d_model). Row ``r`` corresponds to item id ``r + 1``.
        Detached, on CPU.
    """

    # 这里取出模型中“物品作为输入上下文”的嵌入，或者“物品作为预测目标”的输出权重
    if which == "output":
        weight = model.head.weight
    elif which == "input":
        weight = model.tok_emb.weight
    else:
        raise ValueError("which must be 'output' or 'input'")

    # # 去掉第 0 行的 Padding Embedding，并断开梯度、移动到 CPU。
    # 为什么要去掉第一行？因为模型 Vocabulary 里通常约定：item id 0 = PAD，item id 1 = 第一个真实商品，item id 2 = 第二个真实商品
    # 第 0 行不是实际商品，而是 Padding Token。
    # .detach()表示：这个 Tensor 只用于分析，不再参与反向传播。
    emb = weight[1:].detach().cpu()

    if normalize:
        # 计算每个物品向量的 L2 长度。
        # clamp_min 防止某个向量长度为 0 时发生除零错误。
        norms = emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # 将每个物品向量归一化为单位向量，便于计算余弦相似度。
        emb = emb / norms

    return emb



def emb_row(item_id):
    """Map an item id (1..n_items) to its row index in ``item_embeddings``."""
    """
    将原始 item_id 转换成 item_embeddings 返回矩阵中的行号。

    因为 Padding 行已经被删除，所以：
        item_id=1 → row 0
        item_id=2 → row 1
        item_id=3 → row 2
    """
    return item_id - 1
