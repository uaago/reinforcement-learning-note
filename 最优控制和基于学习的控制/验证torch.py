import os
# 防止多进程下 NumPy/PyTorch 底层库自动开启多线程导致严重抢占（RL多进程必备设置）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import platform
import multiprocessing as mp
import numpy as np
import torch

torch.set_num_threads(1)

# ==========================================
# 1. 模拟典型的 RL 连续动力学方程 (13维状态: 位置3 + 速度3 + 四元数4 + 角速度3)
# ==========================================
def dynamics_numpy(state, action):
    # state: (13,), action: (6,) -> 3维推力 + 3维控制力矩
    pos, vel, quat, omega = state[:3], state[3:6], state[6:10], state[10:13]
    force, torque = action[:3], action[3:6]

    # 轨道/重力梯度简化耦合项 (模拟三角函数与非线性计算负载)
    r_norm = np.linalg.norm(pos) + 1e-6
    acc = -3.986e5 * pos / (r_norm ** 3) + force / 100.0 + 0.01 * np.sin(pos)

    # 四元数运动学方程: q_dot = 0.5 * Omega(omega) * q
    wx, wy, wz = omega
    qw, qx, qy, qz = quat
    q_dot = 0.5 * np.array([
        -wx*qx - wy*qy - wz*qz,
         wx*qw + wz*qy - wy*qz,
         wy*qw - wz*qx + wx*qz,
         wz*qw + wy*qx - wx*qy
    ])

    # 欧拉角动力学方程: J * omega_dot = torque - omega x (J * omega)
    J = np.array([15.0, 18.0, 22.0])
    J_omega = J * omega
    omega_dot = (torque - np.cross(omega, J_omega)) / J

    return np.concatenate([vel, acc, q_dot, omega_dot])

def rk4_step_numpy(state, action, dt=0.02):
    k1 = dynamics_numpy(state, action)
    k2 = dynamics_numpy(state + 0.5 * dt * k1, action)
    k3 = dynamics_numpy(state + 0.5 * dt * k2, action)
    k4 = dynamics_numpy(state + dt * k3, action)
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # 四元数归一化
    next_state[6:10] /= (np.linalg.norm(next_state[6:10]) + 1e-8)
    return next_state

# 单个 Worker 进程执行的仿真循环
def worker_rollout(steps):
    rng = np.random.default_rng()
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.01])
    for _ in range(steps):
        action = rng.uniform(-1.0, 1.0, size=(6,))
        # 每个 RL step 通常包含 5 次物理子步积分 (substeps=5)
        for _ in range(5):
            state = rk4_step_numpy(state, action, dt=0.02)
    return state[0]

# ==========================================
# 2. 对照组：PyTorch GPU 向量化 RK4 (直接并行 4096 个环境)
# ==========================================
def benchmark_torch_gpu(num_envs=4096, total_steps=2000):
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    state = torch.tensor([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.01],
                         device=device, dtype=torch.float32).repeat(num_envs, 1)
    J = torch.tensor([15.0, 18.0, 22.0], device=device, dtype=torch.float32)

    def dynamics_torch(s, a):
        pos, vel, quat, omega = s[:, :3], s[:, 3:6], s[:, 6:10], s[:, 10:13]
        force, torque = a[:, :3], a[:, 3:6]
        r_norm = torch.norm(pos, dim=-1, keepdim=True) + 1e-6
        acc = -3.986e5 * pos / (r_norm ** 3) + force / 100.0 + 0.01 * torch.sin(pos)

        wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        q_dot = 0.5 * torch.stack([
            -wx*qx - wy*qy - wz*qz,
             wx*qw + wz*qy - wy*qz,
             wy*qw - wz*qx + wx*qz,
             wz*qw + wy*qx - wx*qy
        ], dim=-1)

        J_omega = J * omega
        omega_dot = (torque - torch.linalg.cross(omega, J_omega)) / J
        return torch.cat([vel, acc, q_dot, omega_dot], dim=-1)

    # 预热 GPU
    action = torch.randn((num_envs, 6), device=device)
    _ = dynamics_torch(state, action)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    dt = 0.02
    for _ in range(total_steps):
        action = torch.rand((num_envs, 6), device=device) * 2 - 1
        for _ in range(5):
            k1 = dynamics_torch(state, action)
            k2 = dynamics_torch(state + 0.5 * dt * k1, action)
            k3 = dynamics_torch(state + 0.5 * dt * k2, action)
            k4 = dynamics_torch(state + dt * k3, action)
            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            state[:, 6:10] = state[:, 6:10] / (torch.norm(state[:, 6:10], dim=-1, keepdim=True) + 1e-8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (num_envs * total_steps) / elapsed

# ==========================================
# 3. 主测试流程
# ==========================================
if __name__ == "__main__":
    print("=" * 55)
    print(f"系统平台: {platform.system()} | 逻辑核心数: {os.cpu_count()}")
    if torch.cuda.is_available():
        print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 55)

    # 测试 1: 单核纯算力 (决定单个环境步进极限速度)
    single_steps = 20000
    t0 = time.perf_counter()
    worker_rollout(single_steps)
    t_single = time.perf_counter() - t0
    fps_single = single_steps / t_single
    print(f"[1] 单核串行性能 (1 Env)   : {fps_single:>10,.0f} steps/sec (耗时 {t_single:.2f}s)")

    # 测试 2: 多核并行 RL 采样 (模拟 SubprocVecEnv 吞吐量)
    for n_workers in [4, 8, 12, 16]:
        if n_workers > os.cpu_count():
            continue
        steps_per_worker = 15000
        t0 = time.perf_counter()
        with mp.Pool(processes=n_workers) as pool:
            pool.map(worker_rollout, [steps_per_worker] * n_workers)
        t_multi = time.perf_counter() - t0
        fps_multi = (n_workers * steps_per_worker) / t_multi
        print(f"[2] {n_workers:>2d} 核并行采样 ({n_workers:>2d} Envs): {fps_multi:>10,.0f} steps/sec (耗时 {t_multi:.2f}s)")

    # 测试 3: PyTorch GPU 向量化并行 (直观感受改写后的提速幅度)
    fps_gpu = benchmark_torch_gpu(num_envs=4096, total_steps=1000)
    if fps_gpu:
        print("-" * 55)
        print(f"[3] PyTorch GPU 向量化 (4096 Envs): {fps_gpu:>10,.0f} steps/sec")
    print("=" * 55)