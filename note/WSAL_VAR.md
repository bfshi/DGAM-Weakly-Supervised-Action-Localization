# Weakly Supervised Action Localization - Variational Method



Data: $\{(x_i, y_i)\}_{i = 1}^{N}$, where $x_i \in \mathbb{R}^{M \times D}$ is feature, $M$ is the length of each video, $D$ is dimension of feature of a single frame, and $y_i$ is label.

We want to predict $\lambda_i \in [0, 1]^M$, which indicates the extent that each of $M$ frames belongs to foreground.

Following the principle of *Maximum A Posteriori (MAP)*, we aim to maximize 
$$
\max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N log P(\lambda_i | x_i, y_i) \\
\Leftrightarrow \max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N log P(x_i, y_i | \lambda_i),
$$
where we omit prior of $\lambda$.

We can further decompose (1) to get
$$
\max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N (log P(x_i | \lambda_i) + log P(y_i | x_i, \lambda_i)).
$$




First, we look at the second term in (2). If we assume that the label is only dependent on foreground frames, i.e., frames with a large $\lambda$, then we can get
$$
log P(y_i | x_i, \lambda_i) = log P(y_i | \lambda_i^T x_i),
$$
which means maximizing the second term in (2) **is equivalent to** optimizing classification results.

Furthermore, if we also assume that background frames, i.e., frames with low $\lambda$, should prevent the video from being classified to any class, then we can add a regularizer:
$$
log P(y_i | \lambda_i^T x_i) + \beta \ logP(y_{bg} | (1 - \lambda_i)^T x_i).
$$




Now turn to the first term in (2). 

Since we don't know $P(x | \lambda)$, we cannot directly optimize w.r.t. $\lambda$. Here we assume that $P(x | \lambda)$ can be represented by some parameter $\theta^\star$, i.e., $P(x | \lambda) = P_{\theta^\star} (x | \lambda)$, where $\theta^\star$ is unknown to us. We further assume that $P_{\theta^\star} (x | \lambda)$ is from parametric family of distributions $P_\theta (x | \lambda)$, e.g., a neural network.

If we make the hypothesis that $\theta^\star$ satisfies the property of maximum likelihood, i.e.,
$$
\max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N log P_{\theta^\star} (x_i | \lambda_i) \geq \max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N log P_{\theta} (x_i | \lambda_i), \quad \forall \theta,
$$
then we can rewrite (2) as 
$$
\max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N (log P_{\theta^\star}(x_i | \lambda_i) + log P(y_i | x_i, \lambda_i)) \\
\Leftrightarrow \max_{\theta} \max_{\lambda_i} \frac{1}{N} \sum_{i = 1}^N (log P_{\theta}(x_i | \lambda_i) + log P(y_i | x_i, \lambda_i)).
$$
Under this formulation, we can optimize w.r.t. $\lambda$ and $\theta$ alternatively. $P_{\theta}(x | \lambda)$ can be modeled by a conditional-VAE.





### 如何解释$P(y | x, \lambda)$ 和 $P(x | \lambda)$ 的关系以及 $P(x | \lambda)$ 的作用？

+ 从聚类的角度解释后者（聚类可以降低entropy）
+ 前者只是解决了单个帧和类别的关系，并没有显式地建模帧与帧之间的关系（或者说帧本身的性质），而后者则会让选中的帧之间相互尽量相似。例如video_test_0001391中的25-27秒，描述的是一个人拿着标枪在地上走（视频的label是扔标枪），这个片段和label是很相关的，所以没有被分为背景，但这个片段的feature和扔标枪的动作（前景）的feature有很大差别，所以在加上$P(x | \lambda)$后就没有被选中。

+ 从generative model 和 discriminative model的角度？？
+ 可以在一个动作片段属于多个类别的时候，让不同类别之间的预测更加consistent？
+ 可以解决action-context separation的问题，因为很多context和对应的类别很相关，但是并不属于前景。
+ 和free energy principle有点类似，都是先去预测，然后再观测，最后根据观测和预测之间的区别（surprise）去更新模型



### 有哪些重要参数可以拿来ablation study？

+ latent dimension（会影响重建以及生成质量）（见Experiments-2019.10.7）
+ CVAE中prior和condition的关联程度（会影响生成质量）（见Experiments-2019.10.23）
+ CVAE中kld_loss和recon_loss的比例（trade-off）（见Experiments-2019.10.7）
+ $l_{recon}$的系数（rgb和flow的系数可以不同）（见Experiments-2019.10.17）
+ $l_{bg}$的系数



### 可以统计哪些数据来支持motivation？

+ 那些分类分数高但前景分数低的帧占的比例，以及分类分数低但前景分数高的帧占的比例



### 还可以添加哪些实验？

+ failure cases