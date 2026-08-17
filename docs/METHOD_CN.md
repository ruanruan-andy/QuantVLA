# QuantVLA-OPQD-v2 方法

## 1. 方法概览

QuantVLA-OPQD-v2 是一个面向量化视觉-语言-动作模型的**在线稀疏蒸馏**方法。它以冻结的全精度策略作为教师，以 W4A8 QuantVLA 作为学生；学生先在目标环境中自主执行，再仅在少量、具有代表性的访问状态上接受教师监督。训练只更新学生 action head 中注意力投影的 LoRA 参数，不更新视觉/语言 backbone、基础 action head 权重、量化参数或教师参数。

方法由四个相互衔接的部分组成：

1. 学生策略产生完整的 on-policy 轨迹；
2. 用教师与学生在同一状态、同一 diffusion 初始噪声下的动作差异定义量化偏差与未来风险；
3. 在四个时间阶段内，以“高分状态 + 随机状态”的固定配额选择 $16$ 个蒸馏状态；
4. 对这些状态进行加权动作蒸馏，并用 clean anchor 抑制目标域适配造成的遗忘。

本文中的 OPQD 特指上述“on-policy 轨迹、四阶段选择、加权蒸馏与 clean anchor”组成的实现。它不是离线行为克隆：监督目标由教师在学生实际访问到的状态上在线生成。

## 2. 记号与模型

设冻结的全精度教师为 $f_T$，量化学生为 $f_S$。学生由固定的量化基础参数 $\theta_Q$ 与可训练 LoRA 参数 $\phi$ 组成，即 $f_S(\theta_Q,\phi)$。教师参数 $\theta_T$、学生基础参数 $\theta_Q$ 与量化配置在训练中均保持冻结，只有 $\phi$ 被优化。

一次学生执行得到长度为 $T$ 的轨迹：

$$
\mathcal{T}=\left\{\left(o_t,\xi_t,\mathbf{A}_t^S,\mathbf{A}_t^T\right)\right\}_{t=0}^{T-1}.
$$

其中：

- $t$ 是环境时间步；
- $o_t$ 是学生在第 $t$ 步实际观察到的多模态状态，包括第三视角图像、腕部图像、机器人状态和语言指令；
- $\xi_t$ 是该时间步生成 diffusion action chunk 时使用的初始高斯噪声；
- $\mathbf{A}_t^S$ 与 $\mathbf{A}_t^T$ 分别是学生和教师的预测 action chunk；
- $T$ 是实际 rollout 长度：任务成功时提前结束，否则在 suite 对应的 horizon 结束。

对同一 $o_t$，教师和学生复用相同的 $\xi_t$。因此二者差异主要来自量化与学生 LoRA 参数，而不会被不同 diffusion 采样噪声混入。

LIBERO 每一步只执行 action chunk 的第一个动作。记该动作的连续控制维数为 $d=7$，包括 $x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw}$ 和 gripper；记 action chunk 中第一个动作分别为 $\mathbf a_t^S\in\mathbb{R}^{d}$ 与 $\mathbf a_t^T\in\mathbb{R}^{d}$。

## 3. Student-on-policy 轨迹采集

每个训练 episode 从目标环境重置后开始。对每个时间步，学生 $f_S$ 根据 $o_t$ 生成 $\mathbf{A}_t^S$，并将第一个动作 $\mathbf a_t^S$ 发送到环境；教师 $f_T$ 在完全相同的 $o_t$ 和 $\xi_t$ 上生成 $\mathbf{A}_t^T$，但不控制环境。环境的后续状态由学生动作决定，因此训练数据来自学生当前策略实际会访问的状态分布。

这种设计使蒸馏目标与部署时的分布一致：如果量化学生在某个偏移状态进入错误区域，后续状态仍会被记录并可成为蒸馏候选。轨迹收集阶段不保留计算图；在状态被选中后，才重新计算其学生 action head 前向并进行反向传播。

## 4. 量化偏差与未来风险

### 4.1 即时量化偏差

对每个时间步，定义学生与教师执行动作之间的均方误差：

$$
q_t=\frac{1}{d}\left\|\mathbf a_t^S-\mathbf a_t^T\right\|_2^2.
$$

其中 $q_t\geq0$ 为第 $t$ 个状态的即时量化偏差；$d=7$ 是动作维数；$\|\cdot\|_2$ 是 Euclidean norm。较大的 $q_t$ 表示学生在当前状态上与全精度教师分歧更大。

### 4.2 折扣未来风险

仅按 $q_t$ 选择状态会偏向单步尖峰。为强调持续性偏差，OPQD 为每一步计算一个截断的未来折扣平均：

$$
r_t=
\frac{\sum_{j=0}^{J_t}\gamma^j q_{t+j}}
     {\sum_{j=0}^{J_t}\gamma^j},
\qquad
J_t=\min\left(H,T-1-t\right).
$$

其中 $r_t$ 是时间步 $t$ 的局部未来风险；$H$ 是最大前瞻偏移，默认 $H=4$，故每个完整窗口包含当前 $q_t$ 与最多 $4$ 个未来项；$\gamma$ 是折扣系数，默认 $\gamma=0.9$；$J_t$ 使窗口在轨迹末尾自动截断。分母对有效窗口重新归一化，避免末尾状态因可见未来步数较少而系统性低分。

## 5. 四阶段、固定配额的状态选择

### 5.1 时间阶段

将长度为 $T$ 的轨迹按相对时间划为 $P=4$ 个阶段：Early、Mid-1、Mid-2 与 Late。零起始索引 $t$ 的阶段编号为：

$$
\pi(t)=\min\left(P-1,\left\lfloor\frac{Pt}{T}\right\rfloor\right).
$$

其中 $\pi(t)\in\{0,1,2,3\}$ 是时间步 $t$ 的阶段编号，$P$ 是阶段数。阶段划分依赖相对进度而不是固定绝对步数，因此可适配成功提前结束或不同 horizon 的轨迹。

### 5.2 阶段内优先级分数

在每一个阶段内部，分别对 $q_t$ 和 $r_t$ 做 percentile rank；相同数值使用平均 rank。记 $\operatorname{rank}_p(x_t)\in[0,1]$ 为状态 $t$ 在阶段 $p$ 内的 percentile rank，则优先级分数为：

$$
s_t=\alpha_q\operatorname{rank}_{\pi(t)}(q_t)
   +\beta_r\operatorname{rank}_{\pi(t)}(r_t).
$$

其中 $s_t$ 是选择分数；$\alpha_q$ 和 $\beta_r$ 分别控制即时偏差与未来风险的重要性，默认均为 $1$。阶段内排序使各阶段的分数尺度可比较，而不会因为后期或前期的绝对误差范围更大而垄断全部预算。

### 5.3 固定配额与时间间隔

对每个阶段，OPQD 选择 $k_p=2$ 个按 $s_t$ 降序的 priority states，以及 $k_r=2$ 个从剩余候选中随机采样的 random states。因此每阶段恰有 $k_p+k_r=4$ 个状态，整条轨迹固定得到：

$$
K=P(k_p+k_r)=4(2+2)=16
$$

个状态。

为避免相邻视频帧产生高度冗余的监督，任意新候选 $i$ 必须与当前已选集合 $\mathcal I$ 满足：

$$
|i-j|\geq g,\qquad \forall j\in\mathcal I.
$$

其中 $g$ 是目标最小时间间隔，默认 $g=4$；$i,j$ 是轨迹索引。选择按阶段顺序进行，且约束对之前阶段已选状态同样生效。

若某个阶段在当前 $g$ 下无法凑齐其 $2+2$ 配额，selector 只在该阶段内依次尝试 $g=4,3,2,1$，直到配额补齐。它不会将名额转移给其他阶段，也不会重复选取同一状态。若轨迹总长度少于 $K=16$，或某一阶段不足以提供其配额，训练直接报错而不是生成不完整样本。

随机状态并非无约束噪声：它们仍需满足阶段配额、去重和有效 gap。其作用是避免始终蒸馏 top-$k$ 高误差状态而造成的选择偏差，并保留不同难度的目标域状态。

## 6. 稀疏加权动作蒸馏

设最终选择的状态索引集合为 $\mathcal I$，满足 $|\mathcal I|=K=16$。对每个 $i\in\mathcal I$，以选中的优先级分数构造截断权重：

$$
w_i=\operatorname{clip}\left(
\frac{s_i}{\max\left(\frac{1}{K}\sum_{j\in\mathcal I}s_j,\epsilon\right)},
w_{\min},w_{\max}
\right),
\qquad
\bar w_i=\frac{w_i}{\sum_{j\in\mathcal I}w_j}.
$$

其中 $w_i$ 是裁剪前归一化后的状态权重，$\bar w_i$ 是用于损失的归一化权重；$\epsilon=10^{-8}$ 防止除零；$w_{\min}=0.5$、$w_{\max}=2.0$ 限制单个状态的影响。权重只由 rollout 后计算的分数确定，在反向传播中视为常数。

对入选状态，代码从采集时保存的 observation 重新计算学生 backbone feature 与 action input；同时复用保存的教师动作和 diffusion 噪声 $\xi_i$。单状态蒸馏损失为：

$$
\ell_i=
\frac{1}{d}\left\|\hat{\mathbf a}_i^S(\xi_i)-\mathbf a_i^T(\xi_i)\right\|_2^2.
$$

其中 $\hat{\mathbf a}_i^S(\xi_i)$ 是带梯度重新计算得到的学生第一个动作；$\mathbf a_i^T(\xi_i)$ 是在同一噪声下缓存的教师第一个动作；$d=7$。主蒸馏目标为：

$$
\mathcal L_{\mathrm{OPQD}}=
\sum_{i\in\mathcal I}\bar w_i\ell_i.
$$

目标域主损失只对 $16$ 个状态反传，而不是对整条轨迹的每一步都反传；但候选评分仍基于完整 student-on-policy 轨迹，因此同时保留了运行时效率与状态覆盖。

实现时，$16$ 个主状态以各自的 $\bar w_i$ 逐个调用反向传播；这与先显式求和得到 $\mathcal L_{\mathrm{OPQD}}$ 再反传在数值上等价。clean anchor 的额外反传见下一节。

## 7. Clean anchor 与总目标

仅用目标域稀疏蒸馏更新 LoRA 可能导致学生偏离原有的 clean-domain 行为。为此，OPQD 额外从 clean LIBERO 环境采集短轨迹，并将其状态缓存到容量为 $R=256$ 的 replay buffer。每次更新从 buffer 中均匀采样至多 $B=4$ 个 anchor 状态。

对 anchor 集合 $\mathcal B$，使用与主蒸馏相同的教师—学生执行动作 MSE：

$$
\mathcal L_{\mathrm{anchor}}
=\frac{1}{|\mathcal B|}
\sum_{b\in\mathcal B}
\frac{1}{d}
\left\|\hat{\mathbf a}_b^S(\xi_b)-\mathbf a_b^T(\xi_b)\right\|_2^2.
$$

其中 $\mathcal B$ 是当前采样的 clean anchor 状态集合；$b$ 是其中一个状态索引；$\xi_b$ 是该状态保存的初始噪声。总训练目标为：

$$
\mathcal L=
\mathcal L_{\mathrm{OPQD}}
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}.
$$

其中 $\lambda_{\mathrm{anchor}}=0.1$ 控制 clean 保持项的强度。clean anchor 是正则化项，不是 validation，也不用于选择最佳 checkpoint；其短 rollout 最多采集 $4$ 步，因此其是否完成任务不具有成功率含义。

实现时，若 replay 非空，则每次 optimizer update 从中采样 $|\mathcal B|=\min(4,|\mathrm{replay}|)$ 个状态，并以 $\lambda_{\mathrm{anchor}}/|\mathcal B|$ 逐个反传。因此它与上式的 $\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}$ 完全等价，同时在主损失之外额外引入至多 $4$ 个 anchor 状态。

## 8. 可训练参数与优化

LoRA 仅插入 action head 注意力模块中的线性 $Q/K/V$ 投影，即名称包含 `action_head` 且包含 `to_q`、`to_k` 或 `to_v` 的 Linear 层。其余模块均冻结，包括语言/视觉 backbone、action head 的基础权重、量化层和全精度教师。

默认 LoRA 配置为 rank $r=16$、缩放系数 $\alpha_{\mathrm{LoRA}}=32$、dropout $0.05$、无 bias。每采集一条 trajectory，对相同的 $K=16$ 个主状态执行 $U=5$ 次 optimizer update；每次 update 都重新从 clean replay 采样 anchor。

优化器为 AdamW，学习率为 $5\times10^{-5}$，weight decay 为 $0.01$。学习率先进行 $50$ 个 step 的线性 warmup，随后采用 cosine decay；梯度范数裁剪阈值为 $1.0$。这些更新只改变 $\phi$。checkpoint 包含 LoRA adapter；`trainer_state.pt` 还保存 optimizer、scheduler、Python sampler、diffusion noise generator、Torch RNG、NumPy RNG 与 task schedule，以支持确定性恢复。

## 9. 单个 episode 的算法流程

给定当前 LoRA 参数 $\phi$，一次 OPQD 更新按以下顺序进行：

1. 用学生 $f_S(\theta_Q,\phi)$ 执行目标环境，记录完整轨迹 $\mathcal T$；教师只在这些已访问状态上推理。
2. 对所有 $t=0,\ldots,T-1$ 计算 $q_t$ 与 $r_t$，并在各时间阶段得到 $s_t$。
3. 在每阶段按固定 $2$ 个 priority 与 $2$ 个 random 配额、带 gap 约束地选出集合 $\mathcal I$。
4. 缓存并重新物化 $\mathcal I$ 中的状态，计算加权损失 $\mathcal L_{\mathrm{OPQD}}$。
5. 采集或复用 clean replay，从 $\mathcal B$ 计算 $\mathcal L_{\mathrm{anchor}}$。
6. 对总目标 $\mathcal L$ 反向传播，裁剪梯度并更新 LoRA 参数 $\phi$；重复 $U=5$ 次。

该过程使状态选择、权重分配和参数更新都由学生当前访问轨迹决定，同时通过显式阶段覆盖、随机补充和 clean anchor 控制稀疏蒸馏的偏差与遗忘。
