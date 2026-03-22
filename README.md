# Diffusion Policy Consistency Exploration

这是一个基于原始 Diffusion Policy 代码库进行学习、整理与实验改造的工程，是一个研究型 / 学习型仓库。

- 代码基础来自原始 Diffusion Policy 项目
- 仓库目标偏向方法理解与工程探索，而不是发布一个已经完善封装的通用库
- 原始开源仓库: <https://github.com/real-stanford/diffusion_policy>
- Consistency model的参考仓库：<https://github.com/quantumiracle/Consistency_Model_For_Reinforcement_Learning>

## 训练成果

在 Push-T lowdim 任务上，目前这组实验的核心结论是：

- Diffusion Policy UNet baseline: `100` inference steps, best `test_mean_score = 0.8677`
- Consistency Policy: `4` inference steps, best `test_mean_score = 0.8672`
- 在分数几乎保持不变的情况下，将推理步数压缩了 `25x`

Left: Diffusion Policy UNet baseline (100 inference steps)  
Right: Consistency Policy (4 inference steps)

![PushT baseline vs consistency](assets/readme/pusht_seed100000_side_by_side.gif)

### 结果图表

Inference steps 与最终性能的关系：

![PushT score vs steps](assets/readme/pusht_score_vs_steps.png)

训练过程中 test score 的变化：

![PushT score vs epoch](assets/readme/pusht_score_vs_epoch.png)

50 个测试 seed 的 reward 排序分布：

![PushT reward sorted](assets/readme/pusht_reward_sorted.png)

高 reward episode 数量对比：

![PushT success thresholds](assets/readme/pusht_success_thresholds.png)

## Training Tip
- 在 Push-T lowdim 上，noise-scale curriculum 对 consistency training 的稳定性非常重要。早期实验中，我直接使用完整的 scale 范围进行训练，优化效果较差，最终性能也明显偏低。后续引入课程学习后，将训练 scale 从 `2` 逐步增加到 `150`，训练过程明显更稳定，最终模型达到了 `test_mean_score = 0.8672`。

- 我还尝试加入了类似 BC 的 reconstruction term 来增强训练稳定性。但在当前这组实验中，这一项没有带来明确的性能提升。就目前仓库中的结果来看，Push-T lowdim 上的最佳 consistency 模型仍然来自 `reconstruction_loss_weight = 0.0` 的设置。若要对这一点下更强结论，还需要更严格的控制变量实验。
