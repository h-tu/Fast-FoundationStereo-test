# CLAUDE.md — Fast-FoundationStereo (fork) + iPhone spatial video depth

## 这个仓库是什么
NVIDIA Fast-FoundationStereo（CVPR 2026, arXiv:2512.11130）的官方推理代码，加上我们自己的目标：
把 iPhone 17 Pro Max 录的 spatial video（MV-HEVC 双目）喂给模型，得到 5 m 内的米制深度视频。
我们的全部背景、实测结论、接口和计划都在 **`docs/iphone_spatial_video_plan.md`**，先读它。

## 目录速览
- `core/` 模型（`foundation_stereo.py::FastFoundationStereo.forward(img_L, img_R, iters, test_mode=True)`，输入 RGB 0-255 `(B,3,H,W)`，H/W 为 32 的倍数，输出左图像素视差）。网络不接收 K/基线。
- `scripts/` 官方 demo / ONNX / TRT / profiling；`scripts/run_demo.py` 有 `cv2.imshow` 和 open3d 窗口，**无显示环境会卡死** — use `--headless 1`.
- `scripts/spatial/` 我们的流水线 (overview and commands: `scripts/spatial/README.md`): `probe_metadata.py` → `extract_views.py` → `check_rectification.py` → `run_sequence.py` → `make_depth_video.py` / `eval_depth.py` → `pose_depth.py` (with field-pose-annotator); many clips: `batch_process.py` + `batch_summary.py`.
- `docs/` 文档与小体积结果（`docs/results/`，单文件 ≤ 15 MB）。
- `vid/` 原始 spatial video（**gitignored**，含 GPS 与人脸，需手工 scp 到机器上）。
- `work/` 抽帧、推理输出（gitignored，可随时删）。`weights/` 权重（gitignored）。`tools/` 本地下载的二进制如静态 ffmpeg（gitignored）。

## 硬性规则
1. 永远不要把 `vid/`、`*.MOV`、`work/`、`weights/*.pth`、`tools/` 加进 git。每次 `git add` 前先 `git status`，提交前 `git diff --cached --stat` 确认没有大文件（> 15 MB 一律不进库）。
2. 每完成一个可验证的步骤就 **更新文档再提交**：`docs/iphone_spatial_video_plan.md` 的 §8 勾选项和 §10 执行日志（日期、机器、命令、关键数字、结论），提交信息用 `step N: <做了什么>`。
3. 代码改动优先放 `scripts/spatial/` 或加命令行开关；改 `core/` 只在必要时，保持向后兼容，并在文档里说明原因。
4. 任何数字（基线、hfov、视差偏移、耗时）以实测为准，写进文档时注明来源命令。

## 环境要点
- 模型只跑 CUDA（硬编码 `.cuda()`、`autocast('cuda')`、`torch.compile`、Triton）。按 `readme.md` 用 Docker（`docker/dockerfile`）或 conda（torch 2.6+cu124，`requirements.txt`）。
- 拆 MV-HEVC 需要 **ffmpeg ≥ 7.1**（`ffmpeg -h decoder=hevc | grep view_ids` 应有输出）。系统 apt 版本通常太老，必要时下载静态构建到 `tools/`。
- 权重：README 的 Google Drive 链接，放 `weights/23-36-37/model_best_bp2_serialize.pth` + 同目录 `cfg.yaml`。
