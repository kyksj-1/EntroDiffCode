# ============================================================================
# Mixed-PDE Dataset for EntroDiff Foundation Model (W5)
#
# 论文对应: §3.5 Foundation Model 多 PDE 训练
# 工程方案: Docs/black/W5_foundation_engineering_plan.md §2.2
#
# 设计哲学 (W5 plan §0.2 可配置性铁律):
#   - 绝不硬编码 PDE 列表; 所有 PDE 名称 / 数据文件 / flux 类型从 YAML 配置传入
#   - 新增 PDE 仅需在 YAML 追加一项, 无任何代码改动
#
# 切分逻辑 (W5 plan §2.2):
#   - 每个 PDE 独立 80/10/10 划分 (复用 BurgersDataset 同款), 保证每个 split 对每个 PDE 都有样本
#   - mode='train' 时拼接所有 PDE 的 train split, 每条记录附 pde_id
#
# 接口契约:
#   __getitem__(idx) → dict {trajectory, ic, x_target, pde_id, pde_name}
#   collate_fn(batch_list) → dict {trajectory, ic, x_target, pde_id, pde_name}
#                            其中 pde_id 是 (B,) long Tensor, pde_name 是 list[str]
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MixedPDEDataset(Dataset):
    """
    跨 PDE 的混合数据集, 用于 Foundation Model 训练.

    Args:
        pdes_config:        YAML 'data.pdes' 字段 (list[dict]).
                            每项必填 {name, data_file, flux_type}; 可选 weight (默认 1.0).
        data_dir:           数据根目录 (一般传 env.data_dir).
        mode:               'train' | 'val' | 'test'  — 与 BurgersDataset 一致 (80/10/10).
        conditioning_type:  'none' | 'ic' — 'ic' 时 get_conditioning() 返回首帧.
        mix_strategy:       'uniform' | 'weighted_by_size'.
                            uniform: 拼接所有 PDE 的 split, DataLoader shuffle 后自然均匀.
                            weighted_by_size: 在 __init__ 按 weight × split_size 重复少数 PDE
                                              的索引, 使 epoch 内每 PDE 等概率被访问.

    数据格式:
        每个 .npy 文件 shape: (N_samples, N_time, N_x).
        与现有 BurgersDataset / generate_data.py 完全对齐.

    扩展性 (W5 plan §0.2):
        新加 PDE 只需在 YAML 追加项. 当前所有 PDE Nx=128, 但 collate_fn 已预留分桶 hook
        (注释中说明: 未来不同 Nx 时按 pde_id 分桶分别 forward).
    """

    def __init__(
        self,
        pdes_config: list[dict],
        data_dir: Path | str,
        mode: str = "train",
        conditioning_type: str = "ic",
        mix_strategy: str = "uniform",
    ) -> None:
        super().__init__()
        # 校验 mode (与 BurgersDataset 一致)
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"未知 mode: {mode}, 必须是 train/val/test")
        # 校验 conditioning (与 BurgersDataset 同款选项)
        if conditioning_type not in {"none", "ic"}:
            raise ValueError(f"未知 conditioning_type: {conditioning_type}")
        # 校验 mix_strategy
        if mix_strategy not in {"uniform", "weighted_by_size"}:
            raise ValueError(f"未知 mix_strategy: {mix_strategy}")
        # 至少一个 PDE
        if not pdes_config:
            raise ValueError("pdes_config 不能为空, 至少需要一个 PDE 配置项")

        self.pdes_config = pdes_config            # 保存原始配置 (用于 get_flux_type / debug)
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.conditioning_type = conditioning_type
        self.mix_strategy = mix_strategy

        # ---- 加载每个 PDE 数据并独立切分 ----
        # 注意: 不硬编码 PDE 名称 (W5 plan §0.2)
        self._per_pde_data: list[np.ndarray] = []   # 每元素 (N_split, N_time, N_x)
        self._per_pde_meta: list[dict] = []         # [{name, flux_type, weight, n_split}, ...]

        for pde_cfg in pdes_config:
            # 必填字段校验
            for required_key in ("name", "data_file", "flux_type"):
                if required_key not in pde_cfg:
                    raise KeyError(
                        f"PDE 配置缺少必填字段 '{required_key}': {pde_cfg}"
                    )

            data_path = self.data_dir / pde_cfg["data_file"]
            if not data_path.exists():
                raise FileNotFoundError(
                    f"PDE '{pde_cfg['name']}' 数据文件不存在: {data_path}"
                )

            # 加载完整数据 (复用 BurgersDataset 切分比例 80/10/10)
            print(f"[MixedPDEDataset] 加载 {pde_cfg['name']} ← {data_path}  mode={mode}")
            data = np.load(str(data_path))  # (N, N_time, N_x) 或 (N, N_time, C, N_x)

            # ---- Step 5: channel_slice 支持 multi-component PDE (e.g., Euler ρ 通道) ----
            channel_slice = pde_cfg.get("channel_slice", None)
            if channel_slice is not None:
                if data.ndim != 4:
                    raise ValueError(
                        f"PDE '{pde_cfg['name']}' channel_slice={channel_slice} 但数据 shape={data.shape} "
                        f"不是 (N, Nt, C, Nx) 4D 张量"
                    )
                print(f"  channel_slice={channel_slice}: {data.shape} → ", end="")
                data = data[:, :, channel_slice, :]   # (N, Nt, C, Nx) → (N, Nt, Nx)
                print(f"{data.shape}")
            n_samples = data.shape[0]
            idx_train_end = int(0.8 * n_samples)
            idx_val_end = int(0.9 * n_samples)

            # 按 mode 切片 (与 BurgersDataset 完全一致, 复现性逐字节对齐)
            if mode == "train":
                split = data[:idx_train_end]
            elif mode == "val":
                split = data[idx_train_end:idx_val_end]
            else:
                split = data[idx_val_end:]

            # weight 缺省 1.0
            weight = float(pde_cfg.get("weight", 1.0))

            self._per_pde_data.append(split)
            self._per_pde_meta.append({
                "name": pde_cfg["name"],
                "flux_type": pde_cfg["flux_type"],
                "weight": weight,
                "n_split": split.shape[0],
                "Nx": split.shape[2],
                "N_time": split.shape[1],
            })
            print(f"    {pde_cfg['name']}: {split.shape}  (Nx={split.shape[2]}, weight={weight})")

        # ---- 构建全局索引映射: global_idx → (pde_id, local_idx) ----
        # 这是 mix_strategy 的实现核心.
        self._index_map: list[tuple[int, int]] = []   # [(pde_id, local_idx), ...]
        if mix_strategy == "uniform":
            # 拼接所有 PDE 的索引, 不做权重 oversample
            # epoch 内不同 PDE 的样本按其原始数量出现 (DataLoader shuffle 后自然均匀)
            for pde_id, meta in enumerate(self._per_pde_meta):
                for local_idx in range(meta["n_split"]):
                    self._index_map.append((pde_id, local_idx))
        else:
            # weighted_by_size: 让每个 PDE 的"有效样本数" ≈ max_n_split × weight
            # 实现方式: 找最大 n_split, 把每个 PDE 的索引重复直到达到 target
            # 这保证小 PDE (n_split 小) 也能被等概率访问 (epoch 长度变大)
            max_n = max(meta["n_split"] for meta in self._per_pde_meta)
            for pde_id, meta in enumerate(self._per_pde_meta):
                target = int(max_n * meta["weight"])
                base_n = meta["n_split"]
                # 整数倍 + 余数, 把 [0..base_n-1] 重复直到 >= target
                # 不打乱顺序 (DataLoader shuffle=True 已保证随机性)
                for k in range(target):
                    self._index_map.append((pde_id, k % base_n))

        # ---- 预先按 conditioning_type 决定是否需要返回 IC 通道 ----
        # (在 __getitem__ 中按需处理; 这里仅 sanity 检查所有 PDE 的 N_time ≥ 2)
        if conditioning_type == "ic":
            for meta in self._per_pde_meta:
                if meta["N_time"] < 2:
                    raise ValueError(
                        f"PDE '{meta['name']}' N_time={meta['N_time']} < 2, "
                        "conditioning='ic' 需要至少 2 个时间步以分别取首帧和末帧"
                    )

        print(f"[MixedPDEDataset] 总样本数 (mix={mix_strategy}, mode={mode}): {len(self._index_map)}")

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: 全局索引 ∈ [0, len(self))

        Returns:
            dict:
                trajectory: (2, Nx) float32  仅倒数 2 帧 (Step 5 修复: 跨 PDE N_time 异质 stack 不齐, 仅保留 [t-1, t] 两帧)
                ic:         (1, Nx)     float32  首帧 (IC); conditioning='none' 时也返回, 由训练脚本决定是否使用
                x_target:   (1, Nx)     float32  末帧 (训练目标) = trajectory[-1]
                pde_id:     int                  PDE 索引 (0..N-1)
                pde_name:   str                  便于 debug / logging
        """
        # 索引映射: global → (pde_id, local)
        pde_id, local_idx = self._index_map[idx]
        data = self._per_pde_data[pde_id][local_idx]   # (N_time, Nx)
        meta = self._per_pde_meta[pde_id]

        trajectory_full = torch.tensor(data, dtype=torch.float32)
        # 首帧 ic, 末帧 x_target (与 train_bvaware.py / BurgersDataset 约定一致)
        ic = trajectory_full[0:1, :].clone()           # (1, Nx)
        x_target = trajectory_full[-1:, :].clone()     # (1, Nx)
        # Step 5 (2026-05-04): 跨 PDE N_time 异质 (e.g., Euler 501, Burgers 101), 完整 trajectory stack 失败
        # 仅保留 time loss 需要的倒数 2 帧 (t-1, t), shape 跨 PDE 统一为 (2, Nx)
        trajectory = trajectory_full[-2:, :].clone()    # (2, Nx)

        return {
            "trajectory": trajectory,             # (2, Nx) — 仅最后两帧 (Step 5 修复)
            "ic": ic,                              # (1, Nx)
            "x_target": x_target,                  # (1, Nx)
            "pde_id": pde_id,                      # int (Python 标量)
            "pde_name": meta["name"],              # str
        }

    def get_flux_type(self, pde_id: int) -> str:
        """
        返回该 pde_id 对应的 flux 类型字符串 (W5-D loss 派遣会用到).

        Args:
            pde_id: PDE 索引
        Returns:
            str  如 'burgers' | 'buckley_leverett'
        """
        if pde_id < 0 or pde_id >= len(self._per_pde_meta):
            raise IndexError(f"pde_id={pde_id} 越界, 仅 {len(self._per_pde_meta)} 个 PDE")
        return self._per_pde_meta[pde_id]["flux_type"]

    def get_pde_name(self, pde_id: int) -> str:
        """便于 logging / eval 时按 ID 查名字."""
        return self._per_pde_meta[pde_id]["name"]

    @property
    def num_pde_types(self) -> int:
        """配置的 PDE 数量 (供 model 构造时设 n_pde_types)."""
        return len(self._per_pde_meta)

    @property
    def pde_names(self) -> list[str]:
        """所有 PDE 名称的有序列表."""
        return [m["name"] for m in self._per_pde_meta]

    # ------------------------------------------------------------------
    # collate_fn: 处理 batch 拼接
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch_list: list[dict]) -> dict:
        """
        Batch 拼接.

        当前实现: 简单 stack (所有 PDE Nx 一致, e.g. 128).

        扩展点 (W5 plan §2.2 collate_fn 注释):
            未来若不同 PDE 的 Nx 不一致, 在此按 pde_id 分桶, 每桶单独 stack
            返回 dict 中改为 list of buckets, 训练脚本按桶 forward.
            目前 (W5 Phase 1) 不需要分桶, 所有 PDE 都 Nx=128.

        Returns:
            dict:
                trajectory: (B, N_time, Nx) float32
                ic:         (B, 1, Nx) float32
                x_target:   (B, 1, Nx) float32
                pde_id:     (B,) long Tensor   ← 关键: 后续 model 与 loss 派遣的索引
                pde_name:   list[str] of length B  ← 便于 logging / debug

        Raises:
            ValueError: 同 batch 内出现不同 Nx (说明 PDE Nx 不一致, 需启用分桶分支)
        """
        # ---- 校验 batch 内 Nx 一致 (W5 Phase 1 假设) ----
        nx_set = {item["x_target"].shape[-1] for item in batch_list}
        if len(nx_set) > 1:
            # 触发分桶逻辑的提示信息 (留待 W5 Phase 2 实现)
            raise ValueError(
                f"batch 内出现多种 Nx: {nx_set}. "
                "W5 Phase 1 假设所有 PDE Nx 一致; 多 Nx 需启用 collate_fn 分桶分支."
            )

        # 简单 stack (Phase 1)
        trajectory = torch.stack([item["trajectory"] for item in batch_list], dim=0)
        ic = torch.stack([item["ic"] for item in batch_list], dim=0)
        x_target = torch.stack([item["x_target"] for item in batch_list], dim=0)
        # pde_id 用 long Tensor (model.pde_embedder 接收 long)
        pde_id = torch.tensor([item["pde_id"] for item in batch_list], dtype=torch.long)
        pde_name = [item["pde_name"] for item in batch_list]   # list[str], 不能 tensor

        return {
            "trajectory": trajectory,
            "ic": ic,
            "x_target": x_target,
            "pde_id": pde_id,
            "pde_name": pde_name,
        }

    def get_conditioning(self, batch: dict) -> Optional[torch.Tensor]:
        """
        与 BurgersDataset.get_conditioning() 风格一致, 但接收 collate 后的 batch dict.

        Args:
            batch: collate_fn 返回的 dict (含 'ic' (B, 1, Nx))
        Returns:
            (B, 1, Nx) float32 或 None
        """
        if self.conditioning_type == "ic":
            return batch["ic"]
        return None
