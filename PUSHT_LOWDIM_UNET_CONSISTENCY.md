# Push-T Lowdim UNet Consistency 代码流程图

这份说明对应这次新增的 consistency 版本 lowdim UNet 链路，重点看这几部分：

- 任务：`Push-T`
- 输入：lowdim 观测
- 策略：`ConsistencyUnetLowdimPolicy`
- 主干网络：`ConditionalUnet1D`

建议按下面这条线读：

`train.py -> train_consistency_unet_lowdim_workspace -> pusht_lowdim 配置 -> pusht_dataset -> consistency_unet_lowdim_policy -> consistency_utils -> conditional_unet1d -> pusht_keypoints_runner`

如果你已经看过 `PUSHT_LOWDIM_UNET_FLOW.md`，可以把这份文档理解成“同一条 Push-T lowdim 数据链路，训练目标和推理采样从 diffusion 改成 consistency”。

## 1. 训练主线

训练入口仍然是 `train.py`。Hydra 根据配置里的 `_target_` 实例化：

- `TrainConsistencyUnetLowdimWorkspace`
- `PushTLowdimDataset`
- `ConsistencyUnetLowdimPolicy`
- `PushTKeypointsRunner`

数据侧和原来的 diffusion lowdim 版本基本一样。每个 batch 仍然是固定长度时间窗：

- `obs: [B, 16, 20]`
- `action: [B, 16, 2]`

默认配置下：

- 前 `2` 步 `obs` 被展平成 `global_cond`
- 整段 `16` 步 `action` 作为训练目标 `trajectory`

和 diffusion 版本最大的区别不在数据，而在 loss：

1. 先用 Karras noise schedule 生成一串 `sigma`
2. 对每个样本随机采样相邻两个噪声尺度 `sigma_1 < sigma_2`
3. 用同一个高斯噪声 `z` 分别构造：
   - `noisy_1 = trajectory + z * sigma_1`
   - `noisy_2 = trajectory + z * sigma_2`
4. 当前模型对 `noisy_2` 做一次 consistency prediction
5. teacher 分支对 `noisy_1` 做一次 prediction
6. 让两者输出尽量一致，得到 consistency loss
7. 再把 `pred_2` 直接和 clean trajectory 做一次 reconstruction loss
8. 最终优化目标是：

   `total_loss = consistency_loss + reconstruction_loss_weight * reconstruction_loss`

这里的 teacher 默认是 `ema_model`。也就是说，训练不是“预测加进去的噪声”，而是“让相邻噪声层级的去噪结果保持一致”。
当前版本额外加了一条 supervised anchor，让 student 输出不会只学到“方向正确”，还能更直接贴近真实 action trajectory。

## 2. 推理主线

推理入口仍然是 `policy.predict_action()`，runner 会持续维护最近 `n_obs_steps=2` 个观测并传入：

```python
obs_dict = {
    'obs': Tensor[B, 2, 20]
}
```

policy 的推理流程是：

1. 对 `obs` 做归一化
2. 取前 `2` 步 observation，展平为 `global_cond`
3. 按 `(B, horizon, action_dim)` 初始化一条高噪声 action trajectory
4. 按采样用的 sigma 序列多次调用 `predict_consistency`
5. 得到整段 action trajectory 后反归一化
6. 按 `oa_step_convention=True` 从整段里切出真正执行的 `n_action_steps`

这里也和 diffusion 版本不同：

- diffusion 是按离散 timestep 反复调用 scheduler，从 `x_t -> x_{t-1}`
- consistency 版本没有 DDPM scheduler
- 它直接在几个噪声尺度上反复做“加噪后再一致化修正”

当前配置里 `num_inference_steps=4`，所以推理时只做很少几次网络前向。这也是 consistency 路线最直接的收益：采样步数明显少于传统 diffusion 反推。

## 3. 关键代码链

### 3.1 入口

文件：`train.py`

核心逻辑没变：

```python
@hydra.main(...)
def main(cfg):
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.run()
```

作用：

- 读取 Hydra 配置
- 实例化 workspace
- 启动训练循环

### 3.2 Workspace

文件：`diffusion_policy/workspace/train_consistency_unet_lowdim_workspace.py`

关键职责：

- 构造 dataset / dataloader / normalizer
- 构造 `ConsistencyUnetLowdimPolicy`
- 构造 optimizer、EMA、lr scheduler
- 在训练时调用 `model.compute_loss(...)`
- 周期性做 validation、rollout、checkpoint

这一层和 `train_diffusion_unet_lowdim_workspace.py` 很像，但多了两点 consistency 专属逻辑：

- `compute_loss()` 返回 `loss + metrics`
- 训练和验证时都可以把 `ema_model` 作为 teacher 传进去

日志里也额外记录了：

- `train_consistency_loss`
- `train_reconstruction_loss`
- `train_total_loss`
- `train_num_scales`
- `train_sigma_1_mean`
- `train_sigma_2_mean`
- `val_consistency_loss`
- `val_reconstruction_loss`

### 3.3 配置

文件：`diffusion_policy/config/train_consistency_unet_lowdim_workspace.yaml`

最关键的 consistency 参数是：

```yaml
policy:
  num_inference_steps: 4
  num_train_scales: 150
  sigma_min: 0.002
  sigma_max: 80.0
  sigma_data: 0.5
  rho: 7.0
  loss_type: l2
  reconstruction_loss_weight: 0.25
  scale_consistency_loss: True
  clip_sample: True
  obs_as_global_cond: True
  oa_step_convention: True
```

这些参数可以直接理解成：

- `num_train_scales`: 训练时把噪声范围切成多少个尺度
- `sigma_min/sigma_max/rho`: Karras 噪声调度参数
- `sigma_data`: consistency 参数化里控制 skip/out 权重
- `num_inference_steps`: 推理实际用几个 sigma 点
- `reconstruction_loss_weight`: supervised reconstruction loss 的权重

同一份配置里，Push-T 任务侧仍然是：

- `obs_dim: 20`
- `action_dim: 2`
- `horizon: 16`
- `n_obs_steps: 2`
- `n_action_steps: 8`

### 3.4 数据集

文件：`diffusion_policy/config/task/pusht_lowdim.yaml`  
文件：`diffusion_policy/dataset/pusht_dataset.py`

这里和原来的 Push-T lowdim UNet 完全复用：

- 数据来自 `data/pusht/pusht_cchi_v7_replay.zarr`
- dataset 负责把 episode 切成长度为 `16` 的时间窗
- 单个样本仍然返回：
  - `obs: [16, 20]`
  - `action: [16, 2]`

如果你已经看过 `PUSHT_LOWDIM_UNET_FLOW.md`，这一部分不需要重复深挖，直接把它看成 consistency 版本沿用了同一套数据组织方式即可。

### 3.5 Policy

文件：`diffusion_policy/policy/consistency_unet_lowdim_policy.py`

这个类是这次新链路的核心。

#### 条件构造

默认只支持 global condition：

```python
global_cond = nobs[:, :self.n_obs_steps].reshape(nobs.shape[0], -1)
```

也就是说，模型拿到的是：

- 输入序列：动作轨迹 `trajectory`
- 条件：前 `2` 步 observation 展平后的向量

当前实现明确不支持：

- `obs_as_local_cond=True`
- `obs_as_global_cond=False`
- `pred_action_steps_only=True`

所以这版设计目标比较聚焦：只覆盖最常见的 Push-T lowdim 全 horizon 动作预测。

#### 一次 consistency prediction 在做什么

核心接口：

```python
def predict_consistency(self, trajectory, sigma, global_cond):
```

内部流程可以概括成：

1. 把 `sigma` 整理成 batch 形状
2. 调 `ConditionalUnet1D(trajectory, sigma, global_cond)`
3. 用 consistency 参数化把网络输出和当前 noisy trajectory 融合
4. 得到 denoised 结果

这里不是单纯输出噪声，也不是 scheduler 决定如何更新，而是直接构造：

- `c_skip`
- `c_out`
- `denoised = c_skip * trajectory + c_out * model_output`

所以 `ConditionalUnet1D` 在这条链里更像“一个带噪声尺度条件的修正器”，而不是 DDPM 语义下的纯噪声预测器。

#### 训练 loss

核心接口：

```python
def compute_loss(self, batch, ema_model=None, num_scales=None):
```

主流程：

1. 归一化 `obs/action`
2. 从 `obs` 取 `global_cond`
3. 生成 Karras sigma 序列
4. 随机采样相邻尺度 `sigma_1, sigma_2`
5. 用同一个 `z` 构造两份不同噪声强度的 `noisy trajectory`
6. 当前模型预测 `pred_2`
7. teacher 预测 `pred_1`
8. 计算 `consistency_error(pred_2, pred_1)`，得到 consistency loss
9. 计算 `consistency_error(pred_2, trajectory)`，得到 reconstruction loss
10. 按 `delta_sigma` 对 consistency loss 做可选缩放
11. 组合成总损失：

    `total_loss = consistency_loss + reconstruction_loss_weight * reconstruction_loss`

也就是说，当前版本的监督信号有两部分：

- 一部分来自“teacher 在更低噪声尺度上的输出”
- 一部分直接来自原始干净 action trajectory

前者负责学 consistency，后者负责给策略一个更直接的 imitation anchor。

### 3.6 Consistency 工具函数

文件：`diffusion_policy/model/consistency/consistency_utils.py`

这个文件负责 3 件事：

- `get_karras_sigmas`: 生成训练用 sigma 网格
- `get_sampling_sigmas`: 生成推理时从大到小的 sigma 序列
- `consistency_error`: 计算 `l1 / l2 / pseudo_huber` 误差

这里的 `get_sampling_sigmas()` 会把训练时的 Karras 序列反转，所以推理是从大噪声往小噪声走。

### 3.7 主干网络

文件：`diffusion_policy/model/diffusion/conditional_unet1d.py`

虽然上层训练目标换成了 consistency，但 backbone 仍然复用现有的 `ConditionalUnet1D`。  
它的输入仍然是：

- 序列张量：这里是 action trajectory
- 一个“噪声尺度/step”嵌入
- `global_cond`

所以可以把这次改动理解成：

- backbone 没换
- 数据也没换
- 主要换的是上层目标函数和采样方式

## 4. 和 diffusion lowdim UNet 的对应关系

如果你想把新老两条链快速对齐，可以直接看这张对照：

- 相同点：
  - 都用 `PushTLowdimDataset`
  - 都用 `ConditionalUnet1D`
  - 都把前 `2` 步 `obs` 展平成 `global_cond`
  - 都输出整段 action trajectory，再切出要执行的几步

- 不同点：
  - diffusion 训练目标是预测噪声
  - consistency 训练目标是让不同噪声层级的去噪结果一致
  - diffusion 推理依赖 DDPM scheduler 多步反推
  - consistency 推理直接在少量 sigma 点上重复调用 `predict_consistency`

所以阅读顺序上，最省力的方法通常是：

1. 先看 `PUSHT_LOWDIM_UNET_FLOW.md`
2. 再只关注 consistency 新增的三个文件：
   - `train_consistency_unet_lowdim_workspace.py`
   - `consistency_unet_lowdim_policy.py`
   - `consistency_utils.py`

## 5. 一句话总结

这条 consistency 链路本质上是：

“沿用 Push-T lowdim 的数据组织、沿用 ConditionalUnet1D 的 backbone，但把 diffusion 的噪声预测 + scheduler 反推，改成了 consistency 的相邻噪声尺度对齐训练、少步数采样，以及额外的 clean trajectory reconstruction anchor。” 
