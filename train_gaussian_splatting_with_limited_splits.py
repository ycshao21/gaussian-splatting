#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModelLimitSplits
from utils.general_utils import safe_state
from utils.logger_utils import training_report, prepare_output_directory, prepare_logger
from utils.network_gui_utils import NetworkGUI
from gaussian_renderer import render

import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import kmeans

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 设置是否使用 GUI 服务器
ENABLE_NETWORK_GUI = False


def train_gaussian_splatting_with_limited_splits(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    num_clusters,
    inside_split_times,
    outside_split_times,
    inside_clone_times,
    outside_clone_times
):
    # 准备输出文件夹和 Tensorboard SummaryWriter
    prepare_output_directory(dataset)
    tb_writer = prepare_logger(dataset.model_path)

    # 3D 高斯模型，给点云中的每个点创建一个 3D gaussian
    gaussians = GaussianModelLimitSplits(dataset.sh_degree)
    # 加载场景，读取数据集和每张图片对应的摄像机的参数
    scene = Scene(dataset, gaussians)

   # 为各组参数创建 optimizer 和 lr_scheduler
    gaussians.training_setup(opt)

    # 加载模型参数
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 背景颜色（0：黑，1：白）
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 准备用时计时器
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    # >>> 限制分裂 >>>
    if num_clusters is not None:
        circles_xyzs, circles_rs = kmeans.getCenterAndR(gaussians.get_xyz.cpu().detach(), num_clusters)  # 获取聚类中心和半径
        max_split_times = {
            "inside": inside_split_times,  # 内部分裂次数
            "outside": outside_split_times,  # 外部分裂次数
        }
        max_clone_times = {
            "inside": inside_clone_times,  # 内部克隆次数，理想状态应该设置为0
            "outside": outside_clone_times  # 外部克隆次数，应当小于外部分裂次数
        }
        split_times = 0  # 目前总共分裂了几次
        # <<< 限制分裂 <<<
    
    # 开始训练
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        NetworkGUI.render(
            dataset,
            opt,
            pipe,
            gaussians,
            background,
            iteration,
        )

        # >>> 开始迭代 >>>
        iter_start.record()

        # 对xyz的学习率进行调整
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            # 将球谐函数的次数增加 1
            gaussians.oneupSHdegree()

        # 从 viewpoint_stack 中随机选择一个图片及其相应的相机视角(内外参)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 根据 3D gaussian 渲染该相机视角的图像
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 计算渲染图像和 Ground Truth 图像之间的 loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()
        # <<< 结束迭代 <<<

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)  # L1 loss
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)  # 总 loss
                tb_writer.add_scalar('train_time', iter_start.elapsed_time(iter_end), iteration)  # 每次迭代的时间

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background),
            )

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and \
                        iteration % opt.densification_interval == 0 and \
                        split_times < max_split_times["outside"]:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        # 限制分裂
                        circles_xyzs,
                        circles_rs,
                        max_split_times,
                        max_clone_times,
                        split_times
                    )
                    split_times += 1
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not os.path.exists(scene.model_path):
                    os.makedirs(scene.model_path)
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
                # if iteration == checkpoint_iterations[-1]:
                #     gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                #     v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                #     np.savez(os.path.join(scene.model_path,"imp_score"), v_list.cpu().detach().numpy()) 


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # parser.add_argument('--gpu_id', type=str, default="0")  # 指定使用的 GPU
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_clusters", type=int, default=20)
    parser.add_argument("--inside_split_times", type=int, default=10)
    parser.add_argument("--outside_split_times", type=int, default=120)
    parser.add_argument("--inside_clone_times", type=int, default=0)
    parser.add_argument("--outside_clone_times", type=int, default=40)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if ENABLE_NETWORK_GUI:
        # [NOTE] 如果 GUI 服务器的 IP 和端口被占用，则无法同时使用两张显卡进行训练
        NetworkGUI.init(args.ip, args.port)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    train_gaussian_splatting_with_limited_splits(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.num_clusters,
        args.inside_split_times,
        args.outside_split_times,
        args.inside_clone_times,
        args.outside_clone_times
    )
    # All done
    print("\nTraining complete.")
