# iPhone Spatial Video → Fast-FoundationStereo：可行性、实测结论与执行计划

> 状态：2026-10-14，本机（macOS, M1 Pro, 无 CUDA）已完成格式解析、几何检查和工具脚本；
> 模型推理尚未运行，需在 CUDA 机器上继续（见 §7）。
> 原始视频在 `vid/`，已通过 `.gitignore` 排除（`vid/`、`*.MOV`、`work/`），**不要**改动这条规则：
> 文件内含 GPS 坐标（`com.apple.quicktime.location.ISO6709`）和可识别人脸。

---

## 1. 目标

用 iPhone 17 Pro Max 录制的 spatial video（MV-HEVC 双目）作为 Fast-FoundationStereo 的输入，
得到每帧的**米制深度图 / 点云**。感兴趣的拍摄对象在 **5 m 以内**。

需要回答的三个问题及结论：

| 问题 | 结论 |
|---|---|
| 能否把左右眼拆出来喂给模型 | 能。ffmpeg ≥ 7.1 原生解 MV-HEVC；本机 9.0.1 已验证 |
| 不同硬件会不会让模型不准 | 模型本身与硬件无关，只要求输入满足**已矫正 + 左右正确**；物理参数只在视差→深度那一步用到 |
| 要不要自己标定 | 不必先做。基线、视场角在文件元数据里（Apple 出厂标定）；但实测发现右眼有**随行号变化的垂直偏移（约 ±2.5 px）**，已用一个仿射修正解决；度量尺度仍需用已知距离物体核一次 |

---

## 2. 模型侧要点（来自论文与代码）

论文：Wen, Dewan, Birchfield, *Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching*, CVPR 2026, arXiv:2512.11130v2。

- 架构：EdgeNeXt-small 蒸馏骨干 → GWC(8 组)+concat 代价体 → NAS 搜出的 3D hourglass（含 4 层 Disparity Transformer）→ softmax 回归初始视差 → 单层 SelectiveConvGRU 迭代 → 凸上采样。14.6 M 参数。
- **接口**（`core/foundation_stereo.py::FastFoundationStereo.forward`）：
  - 输入：`image1, image2`，`(B,3,H,W)` float，RGB，**0–255**，H、W 需为 32 的倍数（用 `core/utils/utils.py::InputPadder` 复制填充）。内部自行做 ImageNet 归一化。
  - 关键字：`iters`（GRU 次数，README 用 8）、`test_mode=True`（只返回最终视差）、`optimize_build_volume='pytorch1'|'triton'`。
  - 输出：`(B,1,H,W)` 视差，单位**左图像素**，正值表示右图中对应点在左边（d = x_L − x_R）。
  - 网络**不接收** K 矩阵、基线；只在后处理中 `depth = fx * baseline / disp`。
- 训练数据：合成数据 + 1.4 M 对 Stereo4D 互联网 VR180 伪标签。论文称在单色 / IR（RealSense）上也可用，即对左右眼光度不一致有一定容忍。
- 输入假设：极线水平、无畸变、不要交换左右。`max_disp` 默认 192（1/4 分辨率上 48）。
- 零样本精度（论文 Table 1）：Middlebury-H BP-2 2.20（教师 1.10，RT-IGEV 7.82）。3090 上 640×480 约 49 ms，TensorRT 21 ms。
- 代码只有推理 / 导出 / profiling，无训练；权重通过 `torch.load` 反序列化整个模块对象，结构由 checkpoint 决定。所有路径硬编码 `.cuda()`、`autocast('cuda')`、`torch.compile`、Triton。

---

## 3. iPhone spatial video 格式（已在本地四段视频上验证）

四段视频 `vid/IMG_3142..3145.MOV`，均为 iPhone 17 Pro Max, iOS 27.2，场景为道路施工（人、卡车、玉米地）。

| 项 | 值 | 来源 |
|---|---|---|
| 容器 / 编码 | QuickTime, HEVC Main, MV-HEVC 两个 view | ffprobe |
| 分辨率 / 帧率 | 1920×1080 @ 29.97 fps, yuv420p, BT.709 SDR | ffprobe |
| 码率 | ≈25 Mbps（两眼合计；第二层只存差分） | ffprobe |
| 基线 `blin` | **19 255 µm = 19.255 mm** | Stereo3D side data `baseline` |
| 水平视场角 `hfov` | **63.4°** | `horizontal_field_of_view = 63400/1000` |
| 视差舒适调整 `dadj` | 200/10000 = 2 % 图宽 = 38.4 px | `horizontal_disparity_adjustment`；**仅播放时用，未烙进像素，不用撤销** |
| 投影 | rectilinear | Spherical Mapping side data |
| 左眼 | ffmpeg `vpos:left` ↔ **view_id 1**；`vpos:right` ↔ view_id 0 | 逐像素比对验证 |
| hero eye | 主摄（Wide）；右眼是超广角裁切放大后"事后对齐"的 | Apple 文档 |
| 两眼时间戳 | 完全相同（同一 pts 出两帧） | ffprobe `-view_ids -1` |

由此导出的针孔参数（主点取图像中心）：

```
fx = fy = W / (2·tan(hfov/2)) = 1920 / (2·tan(31.7°)) = 1554.37 px
cx = 959.5, cy = 539.5
baseline = 0.019255 m
fx·b = 29.93 px·m   →  1 m 处视差 ≈ 30 px, 5 m 处 ≈ 6 px
```

---

## 4. 几何检查结果（块匹配 ZNCC，`scripts/spatial/check_rectification.py`）

对 9 个抽样帧对（4 段视频、不同时刻）做 48×48 块匹配，统计 dy = y_R − y_L 与 d = x_L − x_R：

### 4.1 垂直偏移：存在系统性的**行相关**偏移

| 帧 | dy 中位数 | dy 标准差 | \|dy\|<1 px | 拟合 dy = a·y + b | 仿射后残差 |
|---|---|---|---|---|---|
| 3142 t=10 | +0.87 | 0.90 | 57 % | a=−2.73‰, b=+2.29 | 0.45 |
| 3142 t=60 | +0.93 | 1.25 | 46 % | a=−4.52‰, b=+3.40 | 0.32 |
| 3142 t=110 | +0.88 | 1.48 | 45 % | a=−5.16‰, b=+4.20 | 0.31 |
| 3143 t=5 | +1.04 | 1.31 | 43 % | a=−3.74‰, b=+3.01 | — |
| 3143 t=20 | +0.95 | 1.72 | 36 % | a=−4.85‰, b=+3.47 | 0.83 |
| 3144 t=5 | +0.70 | 1.22 | 53 % | a=−4.02‰, b=+2.93 | 0.38 |
| 3144 t=30 | +0.32 | 1.56 | 45 % | a=−5.32‰, b=+3.47 | 0.49 |
| 3145 t=5 | +0.62 | 1.38 | 51 % | a=−4.23‰, b=+2.98 | 0.75 |
| 3145 t=25 | +0.61 | 1.27 | 50 % | a=−4.64‰, b=+3.49 | 0.33 |

解读：右眼相对左眼在竖直方向有约 **0.4–0.5 % 的尺度差**，零点在 y≈700 附近；画面顶部偏 +2～3 px，底部偏 −1 px。
dy 与 x 几乎无关（斜率 < 0.8‰），所以不是旋转，是超广角裁切放大时的竖直比例 / 主点残差，叠加防抖的逐帧裁切（a、b 在帧间有 ±1‰ 的抖动）。

**修正**：只对右眼做竖直仿射 `y_src = (1+a)·y_dst + b`（水平不动，视差不受影响）。在 3143 t=20 上验证：
dy 标准差 1.72 → 0.60，|dy|<1 px 的比例 36 % → **94 %**，视差中位数 7.42 → 7.45（不变）。
`run_sequence.py --auto_affine N` 每 N 帧自动重新拟合。

模型在 1/4 分辨率上采样代价体，2 px 的竖直误差对应 0.5 px，不修也大概率能出结果，但修正成本低（一次 warpAffine），建议始终开启。

### 4.2 水平方向：无穷远处视差略为负

远景（天空边缘、远处玉米地）块的视差在 **−1.3 ～ −0.1 px**，p5 约 −0.3 ～ −1.3 px。说明两眼存在约 **d_inf ≈ −0.5 ± 0.5 px** 的汇聚偏移，或者水平尺度轻微不一致（现有数据分不开这两种解释）。

影响：度量深度 z = fx·b/(d − d_inf)。在 3 m 处 d≈10 px，0.5 px 的偏置就是 **5 % 深度误差**；1 m 处只有 1.7 %。
所以 **d_inf 必须用已知距离的物体校一次**（§7 步骤 4），然后用 `--dinf` 把右眼整体平移，让无穷远回到 0，也避免把负视差喂给只覆盖 [0, max_disp) 的代价体。

### 4.3 精度预算（1080p, fx=1554, b=19.255 mm）

| 距离 | 视差 | Δz @ 0.25 px | Δz @ 0.5 px |
|---|---|---|---|
| 1 m | 29.9 px | 0.8 cm | 1.7 cm |
| 2 m | 15.0 px | 3.3 cm | 6.7 cm |
| 3 m | 10.0 px | 7.5 cm | 15 cm |
| 5 m | 6.0 px | 21 cm | 42 cm |

Δz = z²·Δd/(fx·b)。结论：3 m 内可靠，3–5 m 可用但要接受十几到几十厘米的不确定度。
**不要用 README 的 `--scale 0.5`**（5 m 处只剩 3 px 视差），直接跑 1920×1088（填充后）。

---

## 5. 其他风险（未量化）

- 右眼来自超广角裁切放大约 2 倍且以低码率差分编码：更软、噪声和色调不同。模型对此有一定鲁棒性，但弱纹理（沥青、天空）可能更不稳。
- 防抖 `cinematicExtendedEnhanced` 逐帧裁切 → hfov 元数据是标称值，实际 fx 可能帧间微变（< 1 %）；元数据 hfov 是否已经是裁切后的值未知，需 §7 步骤 4 的尺度核验兜底。
- 卷帘快门：两个传感器不同，快速运动时行间时序差异可能破坏极线假设。
- HEVC 压缩伪影；README 建议无损 PNG，这里只能接受。
- 远景（> 20 m）视差 < 1.5 px，深度基本不可信，点云请用 `--zfar` 截断。

---

## 6. 工具与接口（`scripts/spatial/`，本机均已跑通）

全部只依赖 ffmpeg ≥ 7.1、numpy、opencv；`run_sequence.py` 的非 dry-run 路径另需仓库环境（torch+CUDA、timm 等）。

### 6.1 `probe_metadata.py` — 读元数据、生成 K.txt
```
python scripts/spatial/probe_metadata.py vid/IMG_3143.MOV [--k_out work/3143/K.txt]
```
stdout 为 JSON：`device, width, height, fps, baseline_m, hfov_deg, disparity_adjust_px, fx, fy, cx, cy, has_gps, …`。
`K.txt` 与 `demo_data/K.txt` 同格式：第 1 行 9 个浮点（K 行主序），第 2 行基线（米）。

### 6.2 `extract_views.py` — 拆左右眼为 PNG 序列
```
python scripts/spatial/extract_views.py vid/IMG_3143.MOV work/3143 --start 5 --duration 2 --stride 15
```
输出：
```
work/3143/left/000000.png    # 1920x1080 RGB, 无损
work/3143/right/000000.png
work/3143/K.txt
work/3143/meta.json          # probe 结果 + frame_time_s[] (帧 i 对应的源时间)
```
眼别由 ffmpeg `-map 0:v:0:vpos:left|right` 决定（读文件自带 SEI），不依赖猜测的 view id。
整段 30 s 视频全帧抽取约 900 对 × 2 × ~2.7 MB ≈ 5 GB，建议先 `--stride`。

### 6.3 `check_rectification.py` — 矫正质量检查
```
python scripts/spatial/check_rectification.py L.png R.png [--json out.json]
python scripts/spatial/check_rectification.py --dir work/3143 --every 10
```
输出 §4 中的全部统计；JSON 里的 `suggested_right_vertical_affine.{a,b}` 可直接传给 `run_sequence.py --right_affine a b`。
Python 接口：`match_patches(L, R) -> (N,5)[xL, yL, d, dy, ncc]`，`summarize(rows, W, H) -> dict`。

### 6.4 `run_sequence.py` — 批量推理（CUDA）
```
python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out \
    --model weights/23-36-37/model_best_bp2_serialize.pth \
    --valid_iters 8 --max_disp 96 --volume triton \
    --auto_affine 10 --dinf -0.5 --pc_every 30 --zfar 8
```
每帧流程：读左右 → 右眼一次 `warpAffine`（竖直仿射 + 水平平移 −dinf，双线性，复制边界）→ 可选 `--scale` → 填充到 /32 → `model.forward(test_mode=True)` → 去填充 → 视差（左图像素）→ 深度（米，`remove_invisible` 逻辑同 `run_demo.py`）。
输出：
```
out/disp/000000.npy      float32, 视差 px（输出分辨率 = 输入分辨率 × scale）
out/depth/000000.npy     float32, 米；无效为 0
out/vis/000000.png       左 | 右(已修正) | 视差 turbo 伪彩
out/cloud/000000.ply     二进制 PLY（x y z r g b），每 pc_every 帧一张，z ≤ zfar
out/run_meta.json        全部参数、K、仿射矩阵、每帧耗时、每次 auto_affine 的 (a, b)
```
`--dry_run` 不导入 torch，用零视差走完整个 I/O 与几何路径（本机已验证）。
`--max_disp 96`：对应最近 0.31 m；需要更近再调回 192。

---

## 7. 在 CUDA 机器上的执行计划

0. 环境：按 README 的 Docker（`docker/dockerfile`）或 pip（torch 2.6 + cu124, `requirements.txt`）。
   确认 `ffmpeg -version` ≥ 7.1（镜像里的 apt ffmpeg 可能太老，必要时下载静态包）。
   下载研究用 checkpoint 到 `weights/23-36-37/`（Google Drive 链接在 README；`cfg.yaml` 必须和 `.pth` 同目录）。
1. 先用 `demo_data/` 跑一次 `scripts/run_demo.py`，确认环境和权重无误（输出应与 `output_docker/disp_vis.png` 一致）。
2. 抽帧：先 `--stride 30`（每秒一对）跑通，再全帧。
3. 推理：上面 §6.4 的命令；`--auto_affine 10`。看 `vis/` 是否合理，关注天空、沥青、远景。
4. **度量尺度核验（必做）**：
   - 找一帧里有已知尺寸 / 已知距离的物体（卡车轮毂直径、车道线宽、或下次录像时放一把卷尺 / A4 纸在 1、2、3 m）；
   - 比较 `depth/*.npy` 与真值，解出 d_inf：`d_inf = d_model − fx·b/z_true`；
   - 用天空 / 远处树木区域的视差均值交叉验证（应 ≈ d_inf，而不是 0）；
   - 把 `--dinf` 设为该值重跑。如果尺度误差仍 > 3 %，再怀疑 fx（防抖裁切），可用同一物体解 fx。
5. 对比实验（每种只需几十帧）：`valid_iters 4 vs 8`、`scale 1.0 vs 0.75`、`max_disp 96 vs 192`、开关 `--auto_affine`，记录精度与耗时（`run_meta.json`）。
6. 如需实时：按 README 导出单文件 ONNX → TensorRT（`scripts/make_single_onnx.py --height 1088 --width 1920`），输入须预归一化。
7. 若 §4.1 的仿射修正不够（残差仍 > 1 px），退路是对**拆出的左右对**用棋盘格做 OpenCV `stereoCalibrate + stereoRectify`，标定的是 Apple 处理后的虚拟相机，可一并解出 fx 与 d_inf。

---

## 8. 本机已验证的事实清单

- [x] ffmpeg 9.0.1 `-view_ids` / `vpos:left|right` 可用；`-vsync` 已被移除，脚本用 `-fps_mode vfr`
- [x] 四段视频元数据一致：baseline 19.255 mm, hfov 63.4°, dadj 2 %, 1080p29.97
- [x] vpos:left == view_id 1（逐像素相等）；且几何上 view 1 的内容偏右，符合"左眼"定义
- [x] 右眼垂直仿射修正有效（§4.1）
- [x] `extract_views.py`、`check_rectification.py`、`run_sequence.py --dry_run` 全流程跑通
- [x] 模型推理（需 CUDA）— 见 §10 步骤 1 起
- [ ] d_inf 与 fx 的度量核验（需真值物体）

### 8.1 CUDA 机器进度（prompt.txt 的步骤编号，详情见 §10）

- [x] 步骤 0 环境与数据自检（Windows 11 + RTX 3090 + venv，ffmpeg 9.0.2 静态包在 `tools/`）
- [x] 步骤 1 权重下载、`run_demo.py --headless 1` 与参考输出一致、640×480 基线耗时
- [ ] 步骤 2 抽帧（stride 30）与几何检查
- [ ] 步骤 3 首次真实推理 + 对比实验
- [ ] 步骤 4 d_inf 远景法估计（真值物体待用户提供）
- [ ] 步骤 5 `make_depth_video.py`，IMG_3143 全帧深度视频
- [ ] 步骤 6 其余三段视频 + 汇总表
- [ ] 步骤 7（可选）TensorRT / 时域中值

---

## 9. 参考

- 论文 arXiv:2512.11130v2；仓库 README / `model_card.md`
- Apple WWDC24 *Build compelling spatial photo and video experiences*（builtInDualWideCamera、hero eye、"should be stereo rectified … not enforced"、dadj 语义）
- Apple *Stereo Video ISOBMFF Extensions*（`vexu` / `blin` / `dadj` / `hfov` 盒定义）
- SVD: Spatial Video Dataset, arXiv:2506.06037（iPhone 16 Pro: 19.2 mm, 63.4°, 超广角事后裁切对齐）
- Apple iPhone 17 Pro Max Tech Specs（主摄 24 mm f/1.78，超广角 13 mm f/2.2 120°，spatial video 1080p30）
- FFmpeg MV-HEVC 解码（7.1+, `view_ids`, view specifiers）

---

## 10. 执行日志（CUDA 机器）

机器：Windows 11 Home 10.0.26200，NVIDIA GeForce RTX 3090 24 GB（WDDM），驱动 610.60（`nvidia-smi` 报 CUDA UMD 13.3）。
**不是 Docker 也不是 conda**：系统只有 Python 3.13.15，没有 conda / docker / git 在 PATH，所以用 venv：

```
python -m venv %USERPROFILE%\venvs\ffs
%USERPROFILE%\venvs\ffs\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
%USERPROFILE%\venvs\ffs\Scripts\python -m pip install -r requirements.txt "triton-windows<3.3"
```

得到 torch 2.6.0+cu124、triton 3.2.0（`triton-windows` 3.2.0.post21）、timm 1.0.30、opencv 5.0.0、open3d 0.20.0。
`xformers` 没装（代码里没有引用）。下面所有 `python` 都指这个 venv 的 python。
git 用 GitHub Desktop 自带的 `%LOCALAPPDATA%\GitHubDesktop\app-3.6.7\resources\app\git\cmd\git.exe`。

### 步骤 0 环境与数据自检（2026-10-14）

- `nvidia-smi`：RTX 3090，驱动 610.60。`torch.cuda.is_available()` = True，`torch.version.cuda` = 12.4。`import timm, omegaconf, triton` 均成功。
- ffmpeg：PATH 里没有。下载 gyan.dev `ffmpeg-release-essentials.zip` 解到 `tools/ffmpeg-9.0.2-essentials_build/bin/`（gitignored）。
  `ffmpeg -version` = 9.0.2；`ffmpeg -h decoder=hevc` 有 `-view_ids` / `-view_ids_available` / `-view_pos_available`。
  后续所有命令前加 `$env:PATH = "$PWD\tools\ffmpeg-9.0.2-essentials_build\bin;$env:PATH"`，脚本不改。
- `vid/IMG_3142..3145.MOV` 四个文件在（403 / 88 / 115 / 101 MB）。
- `probe_metadata.py` 四段结果（命令：`python scripts/spatial/probe_metadata.py vid/IMG_xxxx.MOV`，JSON 存 `work/probe_xxxx.json`）：

  | 文件 | device / iOS | 分辨率 @ fps | nb_frames / 时长 | baseline_m | hfov | dadj px | fx | GPS |
  |---|---|---|---|---|---|---|---|---|
  | 3142 | iPhone 17 Pro Max / 27.2 | 1920×1080 @ 29.97 | 3738 / 124.68 s | 0.019255 | 63.4° | 38.4 | 1554.37 | 有 |
  | 3143 | 同上 | 同上 | 818 / 27.28 s | 同上 | 同上 | 同上 | 同上 | 有 |
  | 3144 | 同上 | 同上 | 1065 / 35.52 s | 同上 | 同上 | 同上 | 同上 | 有 |
  | 3145 | 同上 | 同上 | 931 / 31.05 s | 同上 | 同上 | 同上 | 同上 | 有 |

  与 §3 完全一致，无需改文档。view_ids_present = [0, 1]。
- 遇到的问题：PowerShell 管道给 python 的 stdout 加了 UTF-8 BOM，`python probe.py | python -c json.load` 失败；改为在同一个 python 进程里调用 `probe()` 并写文件。

### 步骤 1 权重与 headless demo（2026-10-14）

- 权重：`python -m gdown --folder <README 的 Google Drive 链接> -O weights`（系统 python 装的 gdown 5.x，`--remaining-ok` 已不存在）。
  得到 `weights/{15-44-51,20-26-39,20-30-48,23-36-37}/{cfg.yaml,model_best_bp2_serialize.pth}` 及 `weights/onnx/…`（整个文件夹约 1 GB，全部 gitignored）。
  `23-36-37/model_best_bp2_serialize.pth` 71.1 MB，`cfg.yaml` 同目录。
- `scripts/run_demo.py` 改动（默认行为不变）：
  - 新增 `--headless 1`：跳过 `cv2.imshow/waitKey` 与 open3d 窗口，文件照常写；
  - 新增 `--volume pytorch1|triton`（默认 pytorch1，与原代码相同）；
  - `os.system('rm -rf … && mkdir -p …')` 在 Windows 上不可用，改为 `shutil.rmtree + os.makedirs`（效果相同）。
- 命令：
  ```
  python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir work/demo_out --remove_invisible 0 --denoise_cloud 1 --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100 --headless 1
  ```
  首次 forward 含 torch.compile / Triton 编译约 10 s；输出 `disp_vis.png, depth_meter.npy, cloud.ply, cloud_denoise.ply`。
- 与 `output_docker/disp_vis.png` 比较（同尺寸 2880×540）：左图面板逐像素相同；视差伪彩面板平均 |Δ| = 0.35 / 255，p99 = 4 / 255，仅 0.14 % 像素差 > 16 —— 即 fp16/编译差异级别，肉眼一致。深度中位数 0.497 m。
- 基线耗时（`python scripts/profile_speed.py --valid_iters 8 --max_disp 192 --warmup 15 --total 45`，640×480，`optimize_build_volume='triton'`，fp16 autocast）：
  **52.9 ms / 帧**（30 次平均；首帧 10.6 s 为编译）。README 的 3090 数字是 49.4 ms，Windows/WDDM 慢约 7 %，合理。
- 结论：Windows + venv + triton-windows 下 Triton 核与 torch.compile 都能跑，不需要 Docker。
