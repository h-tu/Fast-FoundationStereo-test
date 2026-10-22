#!/usr/bin/env python3
"""Run Fast-FoundationStereo over a left/right PNG sequence extracted from an
iPhone spatial video (see extract_views.py) and write disparity / depth / vis.

Must run on a CUDA machine (the model hard-codes .cuda(), autocast('cuda'),
torch.compile and Triton).  `--dry_run` exercises the whole I/O + warping path
without torch, so it can be smoke-tested on a laptop.

Per-frame pipeline
  1. read left/right RGB (uint8, HxWx3)
  2. optional geometric fix of the RIGHT eye in ONE warpAffine (sub-pixel, bilinear):
       vertical : y_src = (1+a)*y_dst + b   (--right_affine a b, from check_rectification.py)
       horizontal: x_src = x_dst - d_inf     (--dinf, so that disparity at infinity becomes 0)
     Left is never touched -> disparity map stays aligned with the left image.
  3. optional --scale (resize both), pad to /32 (InputPadder), model.forward(test_mode=True)
  4. unpad, (un)scale, disparity in LEFT-image pixels at output resolution
  5. depth = fx * baseline / disp   (fx scaled with --scale; K.txt from probe_metadata.py)
  6. write  disp/%06d.npy (float32 px), depth/%06d.npy (float32 m),
            vis/%06d.png (left | right | disparity turbo), optional cloud/%06d.ply

Example (GPU box):
  python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out \
      --model weights/23-36-37/model_best_bp2_serialize.pth --valid_iters 8 --max_disp 96 \
      --right_affine -0.0045 3.4 --dinf -0.5 --pc_every 30

Assemble a preview video afterwards:
  ffmpeg -framerate 2 -i work/3143/out/vis/%06d.png -c:v libx264 -pix_fmt yuv420p work/3143/out/vis.mp4
"""
import argparse, glob, json, os, sys, time
import cv2
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')   # repo root (Utils.py, core/)


def read_k(path):
    with open(path) as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].split())), dtype=np.float64).reshape(3, 3)
    baseline = float(lines[1])
    return K, baseline


def right_warp_matrix(a=0.0, b=0.0, dinf=0.0):
    """dst->src affine for cv2.warpAffine(..., flags=INTER_LINEAR|WARP_INVERSE_MAP)."""
    return np.array([[1.0, 0.0, -dinf],
                     [0.0, 1.0 + a, b]], dtype=np.float64)


def fix_right(img, a, b, dinf):
    if a == 0.0 and b == 0.0 and dinf == 0.0:
        return img
    M = right_warp_matrix(a, b, dinf)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def disp_to_depth(disp, fx, baseline, remove_invisible=True, zmin=0.05):
    disp = disp.astype(np.float32).copy()
    if remove_invisible:   # pixels whose match would fall left of the right image border
        xx = np.arange(disp.shape[1])[None, :]
        disp[(xx - disp) < 0] = np.inf
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = fx * baseline / disp
    depth[~np.isfinite(depth)] = 0
    depth[depth < zmin] = 0
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_dir', required=True, help='from extract_views.py: left/ right/ K.txt meta.json')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--model', default=f'{code_dir}/../../weights/23-36-37/model_best_bp2_serialize.pth')
    ap.add_argument('--valid_iters', type=int, default=8)
    ap.add_argument('--max_disp', type=int, default=96, help='192 in README; 96 is plenty for >=0.35 m at 1080p')
    ap.add_argument('--scale', type=float, default=1.0, help='resize factor before the model (1.0 = 1920x1080)')
    ap.add_argument('--volume', default='pytorch1', choices=['pytorch1', 'triton'])
    ap.add_argument('--right_affine', type=float, nargs=2, default=(0.0, 0.0), metavar=('A', 'B'),
                    help='vertical fix of right eye: y_src=(1+A)*y+B (from check_rectification.py)')
    ap.add_argument('--auto_affine', type=int, default=0, metavar='N',
                    help='re-fit the vertical affine (a,b) from the images every N processed frames '
                         '(CPU block matching, ~1-2 s/frame); overrides --right_affine. 0 = off')
    ap.add_argument('--dinf', type=float, default=0.0, help='disparity measured at infinity (px, usually <=0); '
                    'right eye is shifted so it becomes 0')
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--pc_every', type=int, default=0, help='write a PLY point cloud every N processed frames (0=off)')
    ap.add_argument('--zfar', type=float, default=8.0)
    ap.add_argument('--keep_negative', action='store_true', help='do not clip disparity at 0')
    ap.add_argument('--save_vis', type=int, default=1, help='0: do not write vis/*.png (5760x1080, ~8 MB/frame); for long batch runs')
    ap.add_argument('--dry_run', action='store_true', help='no torch/model: zeros disparity, test I/O + warps')
    ap.add_argument('--backend', default='torch', choices=['torch', 'trt'],
                    help='trt: run a TensorRT engine from make_single_onnx.py + build_trt_engine.py (--engine); '
                         'the engine is fixed-size, so the padded frame must match its input shape')
    ap.add_argument('--engine', default=None, help='.engine file for --backend trt (ImageNet-normalised inputs)')
    args = ap.parse_args()

    lefts = sorted(glob.glob(f'{args.seq_dir}/left/*.png'))[args.start:args.end:args.every]
    assert lefts, f'no frames in {args.seq_dir}/left'
    K, baseline = read_k(f'{args.seq_dir}/K.txt')
    a, b = args.right_affine
    for d in ('disp', 'depth', 'vis', 'cloud'):
        os.makedirs(f'{args.out_dir}/{d}', exist_ok=True)

    model = None; trt_runner = None
    if not args.dry_run:
        import torch, yaml
        from omegaconf import OmegaConf
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE, vis_disparity, depth2xyzmap
        torch.autograd.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        if args.backend == 'trt':
            sys.path.insert(0, f'{code_dir}/..')
            from run_demo_single_trt import SingleEngineTrtRunner
            assert args.engine, '--backend trt needs --engine'
            trt_runner = SingleEngineTrtRunner(args.engine)
            trt_mean = torch.tensor([123.675, 116.28, 103.53], device='cuda').view(1, 3, 1, 1)
            trt_std = torch.tensor([58.395, 57.12, 57.375], device='cuda').view(1, 3, 1, 1)
        else:
            with open(f'{os.path.dirname(args.model)}/cfg.yaml') as f:
                cfg = yaml.safe_load(f)
            cfg['valid_iters'] = args.valid_iters; cfg['max_disp'] = args.max_disp
            model = torch.load(args.model, map_location='cpu', weights_only=False)
            model.args.valid_iters = args.valid_iters
            model.args.max_disp = args.max_disp
            model.cuda().eval()
    else:
        from Utils import vis_disparity, depth2xyzmap  # cv2/numpy only

    meta = dict(vars(args), K=K.tolist(), baseline_m=baseline, n_frames=len(lefts),
                right_warp_matrix=right_warp_matrix(a, b, args.dinf).tolist(), timings_ms=[])
    if args.auto_affine:
        sys.path.insert(0, code_dir)
        from check_rectification import match_patches, summarize
    meta['affine_per_frame'] = {}
    for i, lp in enumerate(lefts):
        rp = os.path.join(os.path.dirname(os.path.dirname(lp)), 'right', os.path.basename(lp))  # Windows-safe
        name = os.path.splitext(os.path.basename(lp))[0]
        img0 = cv2.cvtColor(cv2.imread(lp), cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        if args.auto_affine and i % args.auto_affine == 0:
            g0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY).astype(np.float32)
            g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
            st = summarize(match_patches(g0, g1), g0.shape[1], g0.shape[0])
            if 'error' not in st:
                a, b = st['dy_vs_y_slope_px_per_1000px'] / 1000.0, st['dy_vs_y_intercept']
                meta['affine_per_frame'][name] = [a, b, st['dy_std'], st['residual_std_after_affine']]
                print(f'  auto_affine @ {name}: a={a:+.5f} b={b:+.2f} (dy std {st["dy_std"]:.2f} -> {st["residual_std_after_affine"]:.2f})')
        img1 = fix_right(img1, a, b, args.dinf)
        H0, W0 = img0.shape[:2]
        if args.scale != 1.0:
            img0 = cv2.resize(img0, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, (img0.shape[1], img0.shape[0]), interpolation=cv2.INTER_AREA)
        H, W = img0.shape[:2]

        t0 = time.perf_counter()
        if model is None and trt_runner is None:
            disp = np.zeros((H, W), np.float32)
        else:
            t_l = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
            t_r = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(t_l.shape, divis_by=32, force_square=False)
            t_l, t_r = padder.pad(t_l, t_r)
            if trt_runner is not None:   # engine expects ImageNet-normalised float32 (make_single_onnx.py)
                out = trt_runner({'left_image': ((t_l - trt_mean) / trt_std).contiguous(),
                                  'right_image': ((t_r - trt_mean) / trt_std).contiguous()})['disparity']
            else:
                with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                    out = model.forward(t_l, t_r, iters=args.valid_iters, test_mode=True,
                                        optimize_build_volume=args.volume)
            torch.cuda.synchronize()
            disp = padder.unpad(out.float()).cpu().numpy().reshape(H, W)
        dt = (time.perf_counter() - t0) * 1000
        meta['timings_ms'].append(dt)

        if not args.keep_negative:
            disp = disp.clip(0, None)
        fx = K[0, 0] * args.scale
        depth = disp_to_depth(disp, fx, baseline)

        np.save(f'{args.out_dir}/disp/{name}.npy', disp.astype(np.float32))
        np.save(f'{args.out_dir}/depth/{name}.npy', depth.astype(np.float32))
        if args.save_vis:
            vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
            panel = np.concatenate([img0, img1, vis], axis=1)
            cv2.imwrite(f'{args.out_dir}/vis/{name}.png', cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        if args.pc_every and i % args.pc_every == 0:
            Ks = K.copy(); Ks[:2] *= args.scale
            xyz = depth2xyzmap(depth, Ks).reshape(-1, 3)
            rgb = img0.reshape(-1, 3)
            keep = (xyz[:, 2] > 0) & (xyz[:, 2] <= args.zfar)
            write_ply(f'{args.out_dir}/cloud/{name}.ply', xyz[keep], rgb[keep])
        print(f'[{i+1}/{len(lefts)}] {name}  {dt:.1f} ms  disp med {np.median(disp):.2f} px  '
              f'depth med {np.median(depth[depth>0]) if (depth>0).any() else 0:.2f} m', flush=True)

    meta['mean_ms_after_warmup'] = float(np.mean(meta['timings_ms'][3:])) if len(meta['timings_ms']) > 3 else None
    with open(f'{args.out_dir}/run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'done -> {args.out_dir}')


def write_ply(path, xyz, rgb):
    """Minimal binary PLY writer (no open3d dependency)."""
    n = len(xyz)
    header = (f'ply\nformat binary_little_endian 1.0\nelement vertex {n}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    rec = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'], rec['y'], rec['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec['r'], rec['g'], rec['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii')); f.write(rec.tobytes())


if __name__ == '__main__':
    main()
