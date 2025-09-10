"""
Visualization Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import open3d as o3d
import numpy as np
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    if color is not None:
        color = to_numpy(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(coord) if color is None else color
    )
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def save_bounding_boxes(
    bboxes_corners, color=(1.0, 0.0, 0.0), file_path="bbox.ply", logger=None
):
    bboxes_corners = to_numpy(bboxes_corners)
    # point list
    points = bboxes_corners.reshape(-1, 3)
    # line list
    box_lines = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
    lines = []
    for i, _ in enumerate(bboxes_corners):
        lines.append(box_lines + i * 8)
    lines = np.concatenate(lines)
    # color list
    color = np.array([color for _ in range(len(lines))])
    # generate line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Boxes to: {file_path}")


def save_lines(
    points, lines, color=(1.0, 0.0, 0.0), file_path="lines.ply", logger=None
):
    points = to_numpy(points)
    lines = to_numpy(lines)
    colors = np.array([color for _ in range(len(lines))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_line_set(file_path, line_set)

    if logger is not None:
        logger.info(f"Save Lines to: {file_path}")




import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_sp_tensor(
    st,
    *,
    voxel_size: float = 1.0,     # 该层体素边长（与建网格一致）
    mode: str = "l2",            # "l2" | "l1" | "max" | "channel"
    channel: int = 0,            # mode="channel" 时使用
    normalize: bool = True,      # 是否把取值归一到 [0,1]
    colormap: str = "jet",       # matplotlib 色图名
    center: bool = True,         # 是否把点云平移到中心（仅影响显示）
    max_points: int = None,      # 随机下采样用于快速预览
    save_ply: str = None,        # 可选：保存为 ply 文件
    show: bool = True            # 是否弹出 Open3D 交互窗口
):
    """
    将 spconv.SparseConvTensor 可视化为彩色点云（体素中心着色）。
    - st.indices: [N,4]，顺序 (b, z, y, x)；此处默认只有单 batch
    - st.features: [N,C]
    """
    assert hasattr(st, "indices") and hasattr(st, "features"), "输入必须是 SparseConvTensor。"
    idx = st.indices.detach().cpu().numpy()   # [N,4] (b,z,y,x)
    feat = st.features

    # --- 坐标：体素网格 (z,y,x) -> (x,y,z) 并乘体素尺度 ---
    zyx = idx[:, 1:4].astype(np.float32)      # [N,3]
    xyz = zyx[:, ::-1] * float(voxel_size)    # [N,3]
    if center and xyz.shape[0] > 0:
        xyz = xyz - xyz.mean(axis=0, keepdims=True)

    # --- 计算每点标量取值（用于着色） ---
    feat_f = feat.detach().float()            # 用 float 做聚合更稳
    if mode == "l2":
        vals = (feat_f ** 2).sum(dim=1)
    elif mode == "l1":
        vals = feat_f.abs().sum(dim=1)
    elif mode == "max":
        vals = feat_f.abs().max(dim=1).values
    elif mode == "channel":
        c = int(channel)
        if not (0 <= c < feat_f.shape[1]):
            raise ValueError(f"channel 超出范围：0..{feat_f.shape[1]-1}")
        vals = feat_f[:, c]
    else:
        raise ValueError(f"未知 mode: {mode}")

    vals = vals.cpu().numpy()
    if normalize:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin < 1e-12:
            vals = np.zeros_like(vals, dtype=np.float32)
        else:
            vals = ((vals - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)

    # --- 随机下采样（可选） ---
    if max_points is not None and xyz.shape[0] > max_points:
        sel = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[sel]
        vals = vals[sel]

    # --- 颜色映射 ---
    cmap = plt.get_cmap(colormap)
    colors = cmap(vals)[:, :3].astype(np.float32)

    # --- 生成 Open3D 点云 ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if save_ply is not None:
        o3d.io.write_point_cloud(save_ply, pcd)

    if show:
        o3d.visualization.draw_geometries([pcd])

    return pcd  # 返回 pcd 便于你自行处理/保存


@torch.no_grad()
def visualize_sp_tensor_voxels(
    st,
    feat = None,
    voxel_size: float = 1.0,     # 该层体素边长（与构网一致）
    mode: str = "l2",            # "l2" | "l1" | "max" | "channel"
    channel: int = 0,            # mode="channel" 时使用
    normalize: bool = True,      # 将着色值归一到 [0,1]
    colormap: str = "jet",       # matplotlib 色图名
    max_voxels: int = None,      # 随机下采样体素数量（可选，加速预览）
    center: bool = False,        # 仅用于显示：是否把整体平移到中心
    show: bool = True,            # 是否弹出 Open3D 交互窗口
    save_path = None
):
    """
    将 spconv.SparseConvTensor 以体素网格（彩色小方块）方式可视化。
    - 假设只有单 batch（st.indices[:,0] 全 0）。
    - st.indices: [N,4]，顺序 (b, z, y, x)
    - st.features: [N,C]，每个已占据体素的特征
    """
    assert hasattr(st, "indices") and hasattr(st, "features"), "输入必须是 SparseConvTensor。"
    idx = st.indices.detach().cpu().numpy()   # [N,4] (b,z,y,x)
    if feat is None:
        feat = st.features.detach().float()       # [N,C]
    else:
        feat = feat.detach().float()
    # ---- 计算每个体素的标量值（用于着色）----
    if mode == "l2":
        vals = (feat ** 2).sum(dim=1)
    elif mode == "l1":
        vals = feat.abs().sum(dim=1)
    elif mode == "max":
        vals = feat.abs().max(dim=1).values
    elif mode == "channel":
        c = int(channel)
        if not (0 <= c < feat.shape[1]):
            raise ValueError(f"channel 超出范围：0..{feat.shape[1]-1}")
        vals = feat[:, c]
    else:
        raise ValueError(f"未知 mode: {mode}")
    vals = vals.cpu().numpy()

    # 数值归一化到 [0,1] 便于上色
    if normalize:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
            vals = np.zeros_like(vals, dtype=np.float32)
        else:
            vals = ((vals - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)
    else:
        vals = vals.astype(np.float32)

    # ---- 体素中心坐标（与网格对齐）：(x+0.5, y+0.5, z+0.5) * voxel_size ----
    # indices 是 (b,z,y,x)，显示时用 (x,y,z)
    zyx = idx[:, 1:4].astype(np.float32)                # [N,3]
    xyz = zyx[:, ::-1] + 0.5                            # 体素中心（格点 + 0.5）
    xyz *= float(voxel_size)

    # 可选：随机下采样体素数量（渲染更快）
    if max_voxels is not None and xyz.shape[0] > max_voxels:
        sel = np.random.choice(xyz.shape[0], size=max_voxels, replace=False)
        xyz = xyz[sel]
        vals = vals[sel]

    # 上色（把标量值映射到 RGB）
    cmap = plt.get_cmap(colormap)
    colors = cmap(vals)[:, :3].astype(np.float32)

    # ---- 用“点→体素”的方式生成 VoxelGrid（1 点 ≈ 1 体素，颜色为点色）----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 若仅用于显示美观，可把整体平移到中心（不改变体素大小与相对结构）
    if center and len(pcd.points) > 0:
        ctr = np.asarray(pcd.points).mean(axis=0)
        pcd.translate(-ctr)

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(voxel_size))

    if show:
        o3d.visualization.draw_geometries([vg])
    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pcd)
        print("Saved point cloud to {}".format(save_path))

    return vg  # 返回 VoxelGrid，便于你后续保存/复用


import spconv.pytorch as spconv

@torch.no_grad()
def upsample_one_by_pairs(heat_out: torch.Tensor, pair_bwd: torch.Tensor,
                          reduce: str = "mean", fill_value: float = 0.0) -> torch.Tensor:
    """
    用 spconv 的 pair_bwd 把“低分辨率输出上的热度”映射回“高分辨率输入上的热度”。

    heat_out: [N_out]，低分辨率（当前层输出）上的标量（你的 heat）
    pair_bwd: [K, N_in]，spconvX 的 indice_dict[key].pair_bwd
    reduce:   "mean" | "first" —— 多个 offset 命中时如何归并（stride=2 基本等价）
    返回:     heat_in [N_in]，高分辨率（上一层输入）上的热度
    """
    device = heat_out.device
    pb = pair_bwd.to(device=device, dtype=torch.long)  # [K, N_in]
    K, N_in = pb.shape
    heat_in = torch.full((N_in,), fill_value, dtype=heat_out.dtype, device=device)

    if reduce == "first":
        # 把无效位置设成很大的索引，取 min 当作“第一个”
        sentinel = heat_out.numel()  # 超界索引
        idx = torch.where(pb >= 0, pb, torch.full_like(pb, sentinel))
        idx = idx.min(dim=0).values  # [N_in]
        mask = idx < sentinel
        heat_in[mask] = heat_out[idx[mask]]
        return heat_in

    # 默认：对所有有效 offset 取均值（更稳）
    acc = torch.zeros_like(heat_in)
    cnt = torch.zeros_like(heat_in, dtype=torch.int32)
    for k in range(K):
        idx_k = pb[k]                         # [N_in]
        m = idx_k >= 0
        if m.any():
            acc[m] += heat_out[idx_k[m]]
            cnt[m] += 1
    m = cnt > 0
    heat_in[m] = acc[m] / cnt[m].to(acc.dtype)
    return heat_in


# 1) 计算热度
@torch.no_grad()
def compute_sp_heat(feat, mode: str = "l2", channel: int = 0):
    feat = feat.detach().float()
    if mode == "l2":
        heat = (feat ** 2).sum(dim=1)
    elif mode == "l1":
        heat = feat.abs().sum(dim=1)
    elif mode == "max":
        heat = feat.abs().max(dim=1).values
    elif mode == "channel":
        if not (0 <= channel < feat.shape[1]):
            raise ValueError(f"channel 超出范围 0..{feat.shape[1]-1}")
        heat = feat[:, channel]
    else:
        raise ValueError(f"未知 mode: {mode}")

    return heat.to(torch.float32)

@torch.no_grad()
def lift_heat_with_indice_chain(
    x: spconv.SparseConvTensor,
    heat_low: torch.Tensor,
    keys_in_reverse: list,        # 例如 ["spconv4","spconv3","spconv2","spconv1"]
    reduce: str = "mean",
    fill_value: float = 0.0,
):
    """
    按照 keys_in_reverse 依次用 pair_bwd 把 heat 从最低分辨率抬到最高分辨率。
    要求：heat_low 的顺序与 keys_in_reverse[0] 对应层的“输出索引顺序”一致
         （这通常成立：spconv 不会在同分辨率里重排 active 顺序）
    返回：heat_high（最高分辨率上的热度）
    """
    h = heat_low
    for key in keys_in_reverse:
        assert key in x.indice_dict, f"找不到 indice_key={key} 的映射"
        pair_bwd = x.indice_dict[key].pair_bwd  # [K, N_in_of_prev]
        h = upsample_one_by_pairs(h, pair_bwd, reduce=reduce, fill_value=fill_value)

    heat = h.cpu().numpy()
    vmin, vmax = float(np.nanmin(heat)), float(np.nanmax(heat))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-12:
        heat = np.zeros_like(heat, dtype=np.float32)
    else:
        heat = ((heat - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)
    return heat


@torch.no_grad()
def assign_heat_by_exact_grid_match(
    ref_indices: torch.Tensor,   # 最高分辨率 sp tensor 的 indices [N0,4] (b,z,y,x)
    heat_ref: torch.Tensor,      # [N0] 与上述 indices 同序
    grid_coord: torch.Tensor,    # input_dict["grid_coord"] [M,3]，与你的 coord 一一对应
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    用“整数网格坐标一一匹配”把最高分辨率体素热度按你的点顺序取出。
    无需 KNN / origin / voxel_size。
    """
    import numpy as np
    dev = grid_coord.device
    ref_zyx = ref_indices[:, 1:4].detach().cpu().numpy().astype("int64")
    heat_np = heat_ref.detach().cpu().numpy().astype("float32")
    table = { (int(z),int(y),int(x)): float(v) for (z,y,x), v in zip(ref_zyx, heat_np) }
    gc = grid_coord.detach().cpu().numpy().astype("int64")
    out = np.fromiter((table.get((int(z),int(y),int(x)), fill_value) for z,y,x in gc),
                      dtype=np.float32, count=gc.shape[0])
    return torch.from_numpy(out).to(device=dev, dtype=heat_ref.dtype)


import os
import numpy as np
import open3d as o3d
import matplotlib.cm as cm

def o3d_show_coords_heat(
    coord,                # (N,3) torch.Tensor 或 np.ndarray，单位随你（米/毫米都可）
    heat,                 # (N,)   torch.Tensor 或 np.ndarray，已与 coord 一一对齐
    *,
    cmap: str = "jet",
    vmin: float = None,   # 不传=自动(1%-99%分位)；固定跨样本色域时可手动给
    vmax: float = None,
    clip_pct=(1, 99),     # 自动模式下的分位裁剪
    point_size: float = 2.0,
    bg_color=(1, 1, 1),   # 背景白色
    window_name: str = "heat_high (Open3D)",
    width: int = 1280,
    height: int = 960,
    show: bool = True,    # True: 交互查看；False: 只保存
    save_color_ply: str = None,  # 保存“按 colormap 映射后的彩色”点云
    save_scalar_ply: str = None  # 另存“带真实标量 heat 的 PLY”
):
    # ---- to numpy ----
    try:
        import torch
        if isinstance(coord, torch.Tensor): coord = coord.detach().cpu().numpy()
        if isinstance(heat,  torch.Tensor): heat  = heat.detach().cpu().numpy()
    except Exception:
        pass
    coord = np.asarray(coord, dtype=np.float32)
    heat  = np.asarray(heat,  dtype=np.float32).reshape(-1)
    assert coord.ndim == 2 and coord.shape[1] == 3 and heat.shape[0] == coord.shape[0], \
        f"shape 不匹配：coord={coord.shape}, heat={heat.shape}"

    # ---- 颜色映射（稳健归一化）----
    finite = np.isfinite(heat)
    h = heat[finite]
    if h.size == 0:
        vmin_eff, vmax_eff = 0.0, 1.0
    else:
        if vmin is None or vmax is None:
            p_lo, p_hi = np.percentile(h, clip_pct)
            vmin_eff = float(p_lo) if vmin is None else float(vmin)
            vmax_eff = float(p_hi) if vmax is None else float(vmax)
            if not np.isfinite(vmin_eff) or not np.isfinite(vmax_eff) or vmax_eff <= vmin_eff:
                vmin_eff, vmax_eff = float(h.min()), float(h.max())
        else:
            vmin_eff, vmax_eff = float(vmin), float(vmax)
        if vmax_eff <= vmin_eff:
            c = float(h.mean()); vmin_eff, vmax_eff = c - 0.5, c + 0.5

    t = (heat - vmin_eff) / (vmax_eff - vmin_eff + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    rgb = cm.get_cmap(cmap)(t)[:, :3].astype(np.float32)

    # ---- Open3D 对象 ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # ---- 保存（可选）----
    if save_color_ply:
        os.makedirs(os.path.dirname(save_color_ply) or ".", exist_ok=True)
        o3d.io.write_point_cloud(save_color_ply, pcd, write_ascii=False, compressed=True)
        print(f"[saved] {save_color_ply}")

    if save_scalar_ply:
        # 用 ASCII 写入 x y z heat；如果你装了 plyfile 也可改成二进制
        os.makedirs(os.path.dirname(save_scalar_ply) or ".", exist_ok=True)
        with open(save_scalar_ply, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {coord.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float heat\nend_header\n")
            for i in range(coord.shape[0]):
                f.write(f"{coord[i,0]} {coord[i,1]} {coord[i,2]} {float(heat[i])}\n")
        print(f"[saved] {save_scalar_ply}")

    # ---- 交互可视化（按 S 键可截图到 ./o3d_screenshot.png）----
    if show:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=width, height=height, visible=True)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.background_color = np.asarray(bg_color, dtype=np.float32)
        opt.point_size = float(point_size)

        def _save_png_cb(visu):
            visu.capture_screen_image("o3d_screenshot.png", do_render=True)
            print("[saved] o3d_screenshot.png")
            return False
        vis.register_key_callback(ord('S'), _save_png_cb)

        vis.run()
        vis.destroy_window()

    return {"vmin": vmin_eff, "vmax": vmax_eff}


@torch.no_grad()
def visualize_heat_high_o3d(
    x,                          # 低分辨率的 SparseConvTensor（当前层）
    coord,                      # input_dict["coord"]，与最终 heat_high 一一对齐
    feat=None,
    keys_rev=["spconv4", "spconv3", "spconv2", "spconv1"],              # 例如 ["spconv4","spconv3","spconv2","spconv1"]；None 则自动推断
    heat_mode: str = "l2",      # "l2"|"l1"|"max"|"channel"
    heat_channel: int = 0,
    reduce: str = "mean",       # "mean"|"first"（pair_bwd 多命中时的归并）
    # ----- 可视化参数（传给 o3d_show_coords_heat）-----
    cmap: str = "jet",
    vmin: float = None,
    vmax: float = None,
    clip_pct=(1, 99),
    point_size: float = 2.0,
    show: bool = True,
    save_color_ply: str = None,
    save_scalar_ply: str = None,
):
    """
    一步到位：heat_low(x) -> heat_high(最高分辨率) -> Open3D 点云热力图。

    注意：
      - 假设 lift 后得到的 heat_high 与 coord 的顺序一一对应（你的数据就是这样）。
      - 若你的模型 down 的 indice_key 不同，请把 keys_rev 显式传入。
    """

    if feat is None:
        x_feat = x.features
    else:
        x_feat = feat
    # 2) 计算当前层热度
    heat_low = compute_sp_heat(x_feat, mode=heat_mode, channel=heat_channel)  # [N_low]

    # 3) 用 pair_bwd 链逐级抬到最高分辨率
    heat_high = lift_heat_with_indice_chain(x, heat_low, keys_rev, reduce=reduce)  # [N_high]

    # 4) 可视化（Open3D）
    viz_info = o3d_show_coords_heat(
        coord, heat_high,
        cmap=cmap, vmin=vmin, vmax=vmax, clip_pct=clip_pct,
        point_size=point_size, show=show,
        save_color_ply=save_color_ply, save_scalar_ply=save_scalar_ply
    )

    return heat_high, viz_info