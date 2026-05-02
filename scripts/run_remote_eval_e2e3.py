# ============================================================================
# 远程 Heun 采样评估脚本 — E2 Buckley-Leverett + E3 Euler Sod
# 通过 paramiko SSH 到服务器，上传并执行评估脚本，抓取输出
#
# 用法:
#   python scripts/run_remote_eval_e2e3.py
#
# 依赖: pip install paramiko scipy
# ============================================================================
import paramiko
import sys
import textwrap

HOST = "202.121.181.105"
PORT = 227
USER = "liuyanzhi"
PASSWORD = "HUBRtEeOFC8F0moC"
WORKDIR = "/home/liuyanzhi/ljz/EntroDiffCode"

# ============================================================================
# 服务器端评估脚本 (嵌入为字符串，由 paramiko 上传到 /tmp/ 执行)
# ============================================================================
SERVER_EVAL_SCRIPT = textwrap.dedent(r'''
import os, sys, math, gc, json, re, warnings, glob as glob_mod
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ——— 自动检测项目路径 (兼容服务器不同部署位置) ———
PROJECT_CANDIDATES = [
    '/home/liuyanzhi/ljz/EntroDiffCode',
    '/home/liuyanzhi/ljz/EntroDiffCode/PROJECT/black',
    '/home/liuyanzhi/ljz/EntroDiff',
]
PROJECT_ROOT = None
for cand in PROJECT_CANDIDATES:
    if os.path.isdir(os.path.join(cand, 'src', 'models')):
        PROJECT_ROOT = cand
        break
if PROJECT_ROOT is None:
    print("[FATAL] Cannot find project root with src/models/")
    sys.exit(1)

sys.path.insert(0, PROJECT_ROOT)
# 也加 PROJECT_ROOT/src 直接 import 备用
if os.path.isdir(os.path.join(PROJECT_ROOT, 'src')):
    pass  # 已有 sys.path[0]

import torch
import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance  # 1D W₁ distance

# ——— 路径常量 ———
BASE_DIR = Path('/home/liuyanzhi/ljz/EntroDiffCode')
DATA_DIR = BASE_DIR / 'output' / 'data'
EXP_DIR  = BASE_DIR / 'output' / 'experiments'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NU = 1.0           # VE-SDE unit viscosity (σ²=2ντ)
TAU_MAX = 1.0       # max diffusion time
N_STEPS = 50        # Heun steps

print(f"[eval] PROJECT_ROOT={PROJECT_ROOT}", flush=True)
print(f"[eval] DEVICE={DEVICE}  CUDA_VISIBLE={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"[eval] DATA_DIR={DATA_DIR}  exists={DATA_DIR.exists()}", flush=True)
print(f"[eval] EXP_DIR={EXP_DIR}    exists={EXP_DIR.exists()}", flush=True)

# ========================================================================
# 导入 (需要 PROJECT_ROOT 已在 sys.path 中)
# ========================================================================
from src.data.burgers_dataset import BurgersDataset
from src.models.score_param import StandardScore, BVAwareScore
from src.diffusion.samplers import entrodiff_heun_sampler

# ========================================================================
# 工具函数
# ========================================================================

def detect_model_class(ckpt_path, map_location='cpu'):
    """
    从 checkpoint 的 state_dict keys 自动检测模型类型.
    返回: ('standard', inferred_kwargs) 或 ('bvaware', inferred_kwargs)
    """
    sd = torch.load(str(ckpt_path), map_location=map_location)
    keys = list(sd.keys())

    if any('phi_sm_net' in k for k in keys):
        # BVAwareScore
        # 推断 dim: 从 phi_sm_net.enc1.0.weight 的 out_channels 推断
        kwargs = {'return_denoiser': True}
        for k in keys:
            if 'phi_sm_net.enc1.0.weight' in k:
                kwargs['dim'] = sd[k].shape[0]
                break
        # 推断 out_channels: 从 phi_sm_net.final.weight
        for k in keys:
            if 'phi_sm_net.final.weight' in k:
                kwargs['out_channels'] = sd[k].shape[0]
                break
        # 推断 in_channels: 从 phi_sm_net.inc.weight
        for k in keys:
            if 'phi_sm_net.inc.weight' in k:
                kwargs['in_channels'] = sd[k].shape[1]
                break
        return 'bvaware', kwargs
    elif any('net.inc.weight' in k or 'net.enc1.0.weight' in k for k in keys):
        # StandardScore (UNet1D internally has 'inc', 'enc1', etc. but wrapped in 'net.')
        kwargs = {}
        for k in keys:
            if 'net.inc.weight' in k:
                kwargs['in_channels'] = sd[k].shape[1]
                break
        return 'standard', kwargs
    else:
        # Fallback: try both and see which has more matching parameters?
        return 'unknown', {}

def compute_metrics(gen_arr, gt_arr):
    """计算平均 W₁ 和平均 L¹ 相对误差."""
    w1s, l1s = [], []
    gen_arr = np.asarray(gen_arr, dtype=np.float64)
    gt_arr  = np.asarray(gt_arr, dtype=np.float64)
    for i in range(len(gen_arr)):
        g = gen_arr[i].reshape(-1)
        t = gt_arr[i].reshape(-1)
        w1s.append(wasserstein_distance(t, g))
        l1s.append(np.linalg.norm(g - t, 1) / (np.linalg.norm(t, 1) + 1e-10))
    return float(np.mean(w1s)), float(np.mean(l1s))

def load_model_from_ckpt(ckpt_path, device):
    """自动检测模型类别并加载."""
    mtype, kwargs = detect_model_class(ckpt_path, map_location=device)
    if mtype == 'bvaware':
        print(f"    [detect] BVAwareScore kwargs={kwargs}", flush=True)
        model = BVAwareScore(**kwargs).to(device)
    elif mtype == 'standard':
        print(f"    [detect] StandardScore kwargs={kwargs}", flush=True)
        model = StandardScore(**kwargs).to(device)
    else:
        raise RuntimeError(f"Cannot detect model type from ckpt keys")
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    return model, kwargs

# ========================================================================
# E2 Buckley-Leverett 评估
# ========================================================================
BL_DATA = DATA_DIR / "bl_1d_N5000_Nx128.npy"
# fallback naming
if not BL_DATA.exists():
    alt = DATA_DIR / "bl_1d.npy"
    if alt.exists():
        BL_DATA = alt

E2_CKPTS = {
    "e2_baseline": EXP_DIR / "e2_bl_baseline" / "entrodiff_e2bl_base_20260430_170845_ep50.pt",
    "e2_ours":     EXP_DIR / "e2_bl_run"      / "entrodiff_e2_bl_run_20260429_224149_ep200.pt",
    "e2_bvaware":  EXP_DIR / "e2_bvaware_run"  / "entrodiff_e2_bvaware_run_20260430_210312_ep200.pt",
}

print("\n" + "=" * 60, flush=True)
print("  E2 Buckley-Leverett 评估", flush=True)
print("=" * 60, flush=True)

if not BL_DATA.exists():
    print(f"[E2] 数据文件不存在: {BL_DATA}", flush=True)
    print("E2_RESULTS:", flush=True)
    print("E2_END", flush=True)
else:
    print(f"[E2] 加载 BL 数据: {BL_DATA}", flush=True)
    # BL 数据与 Burgers 格式相同: [N, Nt, Nx]
    bl_ds = BurgersDataset(str(BL_DATA), mode='test', conditioning_type='ic')
    bl_raw = bl_ds.data  # numpy [N_test, Nt, Nx]
    n_test_bl = min(len(bl_raw), 500)
    bl_gt = bl_raw[:n_test_bl, -1, :]       # 末帧 ground truth
    bl_ic = bl_raw[:n_test_bl, 0, :]        # 首帧 IC
    nx_bl = int(bl_gt.shape[1])
    print(f"    test samples={n_test_bl}  Nx={nx_bl}  gt_shape={bl_gt.shape}", flush=True)

    ic_tensor = torch.tensor(bl_ic, device=DEVICE).unsqueeze(1)  # [n, 1, Nx]
    gen_shape = (n_test_bl, 1, nx_bl)

    def eval_one_e2(label, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            # 精确路径不存在 → glob 回退找最新 ckpt
            parent = ckpt_path.parent
            if parent.exists():
                pts = sorted(glob_mod.glob(str(parent / "*.pt")),
                             key=lambda p: int(re.search(r'_ep(\d+)', p).group(1))
                                         if re.search(r'_ep(\d+)', p) else -1,
                             reverse=True)
                if pts:
                    ckpt_path = Path(pts[0])
                    print(f"    [fallback] using {ckpt_path.name}", flush=True)
                else:
                    print(f"  {label}: ckpt NOT FOUND (dir={parent})", flush=True)
                    return None, None
            else:
                print(f"  {label}: dir NOT FOUND ({parent})", flush=True)
                return None, None

        print(f"    loading {ckpt_path.name} ...", flush=True)
        try:
            model, kw = load_model_from_ckpt(ckpt_path, DEVICE)
        except Exception as e:
            print(f"    [ERROR] load failed: {e}", flush=True)
            return None, None

        with torch.no_grad():
            gen = entrodiff_heun_sampler(
                model=model, shape=gen_shape,
                sigma_min=0.002,
                sigma_max=math.sqrt(2.0 * NU * TAU_MAX),
                tau_max=TAU_MAX, nu=NU, num_steps=N_STEPS,
                device=DEVICE, zeta_pde=0.0, ic=ic_tensor,
            ).squeeze().cpu().numpy()  # (n, Nx)

        w1, l1 = compute_metrics(gen, bl_gt)
        print(f"  {label}: W1={w1:.4f}  L1={l1:.4f}", flush=True)
        del model; gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return w1, l1

    print("E2_RESULTS:", flush=True)
    for label, ckpt_path in E2_CKPTS.items():
        w1, l1 = eval_one_e2(label, ckpt_path)
        if w1 is not None:
            print(f"{label}, W1={w1:.3f}, L1={l1:.3f}", flush=True)
        else:
            print(f"{label}, W1=N/A, L1=N/A", flush=True)
    print("E2_END", flush=True)

# ========================================================================
# E3 Euler Sod 评估
# ========================================================================
EULER_CANDIDATES = [
    DATA_DIR / "euler_sod_1d_N5000_Nx128.npy",
    DATA_DIR / "euler_sod_N5000_Nx128.npy",
    DATA_DIR / "euler_sod_N2000_Nx256.npy",
    DATA_DIR / "euler_sod_1d.npy",
]
EULER_DATA = None
for path in EULER_CANDIDATES:
    if path.exists():
        EULER_DATA = path
        break

E3_CKPT = EXP_DIR / "e3_euler_run"
# 尝试精确文件名，失败则 glob 回退 (re, glob_mod 已在顶部导入)
e3_ckpt_path = E3_CKPT / "entrodiff_e3_euler_run_20260502_152258_ep200.pt"
if not e3_ckpt_path.exists() and E3_CKPT.exists():
    pts = sorted(glob_mod.glob(str(E3_CKPT / "*.pt")),
                 key=lambda p: int(re.search(r'_ep(\d+)', p).group(1))
                             if re.search(r'_ep(\d+)', p) else -1,
                 reverse=True)
    if pts:
        e3_ckpt_path = Path(pts[0])

print("\n" + "=" * 60, flush=True)
print("  E3 Euler Sod 评估", flush=True)
print("=" * 60, flush=True)

if EULER_DATA is None or not EULER_DATA.exists():
    print(f"[E3] Euler 数据文件不存在 (checked: {[str(p) for p in EULER_CANDIDATES]})", flush=True)
    print("E3_RESULTS:", flush=True)
    print("E3_END", flush=True)
elif not e3_ckpt_path.exists():
    print(f"[E3] ckpt 不存在: {e3_ckpt_path}", flush=True)
    print("E3_RESULTS:", flush=True)
    print("E3_END", flush=True)
else:
    print(f"[E3] Euler data: {EULER_DATA}", flush=True)
    euler_raw = np.load(str(EULER_DATA))
    print(f"    raw shape: {euler_raw.shape}  dtype={euler_raw.dtype}", flush=True)

    # 自动检测数据格式并转换为统一格式: [N, n_comp, Nt, Nx]
    ndim = euler_raw.ndim
    n_comp = 3  # 默认
    if ndim == 4:
        s = euler_raw.shape
        # 可能: (N, 3, Nt, Nx)  or  (N, Nt, Nx, 3)  or  (N, Nt, 3, Nx)
        if s[1] == 3:  # (N, 3, Nt, Nx)
            # Already correct format
            print(f"    format: (N=3, Nt, Nx) ✓", flush=True)
            pass  # 无需转换
        elif s[3] == 3:  # (N, Nt, Nx, 3)
            print(f"    format: (N, Nt, Nx, 3) → permuting", flush=True)
            euler_raw = np.transpose(euler_raw, (0, 3, 1, 2))  # → (N, 3, Nt, Nx)
        elif s[2] == 3:  # (N, Nt, 3, Nx)
            print(f"    format: (N, Nt, 3, Nx) → permuting", flush=True)
            euler_raw = np.transpose(euler_raw, (0, 2, 1, 3))  # → (N, 3, Nt, Nx)
        else:
            print(f"    [WARN] unknown 4D format {s}, treating as (N, 3, Nt, Nx)", flush=True)
    elif ndim == 3:
        print(f"    [WARN] 3D data {euler_raw.shape}, assuming single-component. Treat as (N, Nt, Nx).", flush=True)
        n_comp = 1
        # Wrap to 4D: insert component dim
        euler_raw = euler_raw[:, np.newaxis, :, :]  # (N, 1, Nt, Nx)
    else:
        print(f"[FATAL] Unexpected ndim={ndim}", flush=True)
        sys.exit(1)

    print(f"    final shape: {euler_raw.shape}", flush=True)

    # 80/10/10 test split
    n_total = euler_raw.shape[0]
    idx_test = int(0.9 * n_total)
    euler_test = euler_raw[idx_test:]  # (N_test, n_comp, Nt, Nx)
    n_test_e3 = min(len(euler_test), 500)
    n_comp = euler_test.shape[1]
    Nt_e3 = euler_test.shape[2]
    nx_e3 = euler_test.shape[3]

    # GT: 末帧 (n_comp, Nx); transpose to (n_test, n_comp, Nx)
    euler_gt = euler_test[:n_test_e3, :, -1, :]   # (n_test, n_comp, Nx)
    euler_ic = euler_test[:n_test_e3, :, 0, :]    # (n_test, n_comp, Nx)
    print(f"    test samples={n_test_e3}  n_comp={n_comp}  Nx={nx_e3}", flush=True)

    # 加载模型 (自动检测 BVAwareScore vs StandardScore)
    print(f"    loading ckpt: {e3_ckpt_path}", flush=True)
    try:
        model_e3, kw_e3 = load_model_from_ckpt(e3_ckpt_path, DEVICE)
    except Exception as e:
        print(f"    [ERROR] load failed: {e}", flush=True)
        print("E3_RESULTS:", flush=True)
        print("E3_END", flush=True)
        raise

    # 确定 out_channels
    out_channels = kw_e3.get('out_channels', 1)
    in_channels  = kw_e3.get('in_channels', n_comp * 2)
    print(f"    model: out_channels={out_channels} in_channels={in_channels}", flush=True)

    # 构建 conditioning: IC 张量 [n_test, n_comp, Nx]
    ic_tensor_e3 = torch.tensor(euler_ic, device=DEVICE)

    # 如果模型输出通道与数据通道不同 (e.g. StandardScore out=1 vs n_comp=3)
    # 只评估能匹配的通道
    eval_components = min(out_channels, n_comp)
    gen_shape_e3 = (n_test_e3, eval_components, nx_e3)

    print(f"    sampling shape={gen_shape_e3} ...", flush=True)
    with torch.no_grad():
        gen_e3 = entrodiff_heun_sampler(
            model=model_e3, shape=gen_shape_e3,
            sigma_min=0.002,
            sigma_max=math.sqrt(2.0 * NU * TAU_MAX),
            tau_max=TAU_MAX, nu=NU, num_steps=N_STEPS,
            device=DEVICE, zeta_pde=0.0,
            conditioning=ic_tensor_e3 if in_channels > n_comp else None,
        ).cpu().numpy()  # (n_test, eval_comp, Nx)

    # 逐组件评估
    comp_names = {0: "rho", 1: "u", 2: "p", 3: "E"}
    print("E3_RESULTS:", flush=True)
    for c in range(eval_components):
        # 取对应组件
        c_gt  = euler_gt[:n_test_e3, c, :]   # (n_test, Nx)
        c_gen = gen_e3[:n_test_e3, c, :]      # (n_test, Nx)
        w1, l1 = compute_metrics(c_gen, c_gt)
        cname = comp_names.get(c, f"comp{c}")
        print(f"e3_ours_{cname}, W1={w1:.3f}, L1={l1:.3f}", flush=True)

    # 如果有超出 eval_components 的组件 (e.g. model 只输出 1 通道但数据有 3)
    # 标记为 N/A
    for c in range(eval_components, n_comp):
        cname = comp_names.get(c, f"comp{c}")
        print(f"e3_ours_{cname}, W1=N/A, L1=N/A  # model out_channels={out_channels} < n_comp={n_comp}", flush=True)

    print("E3_END", flush=True)
    del model_e3; gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

print("\n[eval] ALL DONE", flush=True)
''')


def main():
    """主入口: paramiko SSH → 上传评估脚本 → 执行 → 返回输出."""
    print(f"[local] 连接 {USER}@{HOST}:{PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            HOST, port=PORT,
            username=USER, password=PASSWORD,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
    except Exception as e:
        print(f"[local] SSH 连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    print("[local] 已连接. 上传评估脚本到服务器 /tmp/eval_e2e3.py ...")
    try:
        sftp = ssh.open_sftp()
        with sftp.file('/tmp/eval_e2e3.py', 'w') as f:
            f.write(SERVER_EVAL_SCRIPT)
        sftp.chmod('/tmp/eval_e2e3.py', 0o755)
        sftp.close()
    except Exception as e:
        print(f"[local] SFTP 上传失败: {e}", file=sys.stderr)
        ssh.close()
        sys.exit(1)

    cmd = (
        f"cd {WORKDIR} && "
        f"CUDA_VISIBLE_DEVICES=2 python /tmp/eval_e2e3.py"
    )
    print(f"[local] 执行: {cmd}")

    # 使用 exec_command, 设置较长超时
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=3600, get_pty=True)
    # get_pty=True 可以获得更好的流式输出 (stderr 合并到 stdout)

    # 流式读取输出
    print("=" * 60)
    print("[remote output begin]")
    print("=" * 60)

    # 非阻塞轮询读取输出
    channel = stdout.channel
    channel.settimeout(3600)  # 1 hour timeout for long eval

    while not channel.closed or channel.recv_ready() or channel.recv_stderr_ready():
        if channel.recv_ready():
            data = channel.recv(4096)
            if data:
                text = data.decode('utf-8', errors='replace')
                sys.stdout.write(text)
                sys.stdout.flush()
        if channel.recv_stderr_ready():
            data = channel.recv_stderr(4096)
            if data:
                text = data.decode('utf-8', errors='replace')
                sys.stderr.write(text)
                sys.stderr.flush()
        # 检查是否完成
        if channel.exit_status_ready():
            # 清空剩余数据
            while channel.recv_ready():
                data = channel.recv(4096)
                if data:
                    sys.stdout.write(data.decode('utf-8', errors='replace'))
                    sys.stdout.flush()
            while channel.recv_stderr_ready():
                data = channel.recv_stderr(4096)
                if data:
                    sys.stderr.write(data.decode('utf-8', errors='replace'))
                    sys.stderr.flush()
            break

    exit_code = channel.recv_exit_status()
    print(f"\n[local] 远程脚本退出码: {exit_code}")

    ssh.close()
    print("[local] 完成.")


if __name__ == "__main__":
    main()
