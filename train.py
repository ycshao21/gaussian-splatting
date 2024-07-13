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
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
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
USE_NETWORK_GUI = False

USE_EARLY_STOP = False

def training(dataset, opt, pipe, checkpoint_iterations, checkpoint, debug_from):
    # 准备输出文件夹和 Tensorboard SummaryWriter
    prepare_output_directory(dataset)
    tb_writer = prepare_logger(dataset.model_path)

    # 3D 高斯模型，给点云中的每个点创建一个 3D gaussian
    gaussians = GaussianModel(dataset.sh_degree)
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

    # 准备用时测试
    train_start = torch.cuda.Event(enable_timing = True)
    train_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    best_psnr = 0
    patience = 10
    patience_counter = 0

    validation_interval = 120  # 初始评估间隔
    last_validation_iteration = 7000

    eval_start = 7000  # 从第7000轮开始评估
    eval_end = 30000  # 在第30000轮结束评估

    # 开始训练
    first_iter += 1

    # circles_xyzs = [torch.tensor([0.326419, 0.892643, 0.72607], dtype=torch.float64, device="cuda"),
    #                 torch.tensor([5.01227, 1.98988, -0.380355], dtype=torch.float64, device="cuda"),
    #                 torch.tensor([0.662707, -1.07687, 7.05663], dtype=torch.float64, device="cuda")]
    # circles_rs = [2.5, 3, 3]
    circles_xyzs, circles_rs = kmeans.getCenterAndR(gaussians.get_xyz.cpu().detach(), 20)
    max_split_times = {"inside": 5, "outside": 20}
    split_times = 0  # 目前总共分裂了几次

    for iteration in range(first_iter, opt.iterations + 1):        

        if USE_NETWORK_GUI:
            # 为 GUI 渲染当前图像
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

        # >>> 开始迭代 >>>
        train_start.record()

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

        train_end.record()
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
                tb_writer.add_scalar('train_time', train_start.elapsed_time(train_end), iteration)  # 每次迭代的时间

            # 验证
            if USE_EARLY_STOP:
                if eval_start <= iteration <= eval_end:
                    if iteration - last_validation_iteration >= validation_interval:
                        l1_test, psnr_test = evaluate(tb_writer, iteration, l1_loss, scene, render, (pipe, background))
                        last_validation_iteration = iteration

                        # 早停机制
                        if psnr_test > best_psnr:
                            best_psnr = psnr_test
                            patience_counter = 0
                            validation_interval = min(validation_interval * 1.5, 1200)  # 增加评估间隔

                            print(f"\n[ITER {iteration}] Saving Gaussians")
                            scene.save(iteration)
                        else:
                            patience_counter += 1
                            validation_interval = max(validation_interval * 0.8, 120)  # 减少评估间隔
                            if patience_counter >= patience:
                                print(f"Early stopping triggered after {iteration} iterations due to no improvement in psnr")
                                break

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,
                                                circles_xyzs, circles_rs, max_split_times, split_times)
                    split_times += 1
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))
                

def prepare_output_directory(args):    
    """
    设置模型保存路径，并创建输出文件夹

    Parameters
    ----------
    model_path : _type_
        _description_
    """
    # 如果没有指定模型保存路径，则生成一个 10 位的 uuid 作为文件夹名
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("output", unique_str[0:10])
        
    print(f"Output folder: {args.model_path}")

    # 创建输出文件夹
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def prepare_logger(model_path):
    """
    设置 Tensorboard 日志

    Parameters
    ----------
    model_path : str
        模型保存路径

    Returns
    -------
    tb_writer : SummaryWriter
        Tensorboard 的日志记录器
    """
    tb_writer = None 

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")

    return tb_writer


def evaluate(tb_writer, iteration, l1_loss, scene : Scene, renderFunc, renderArgs):
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                        {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            
            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if tb_writer and (idx < 5):
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                    if iteration == 7000:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()

            l1_test /= len(config['cameras'])          
            psnr_test /= len(config['cameras'])
            
            # print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)
        
    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

    return l1_test, psnr_test


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    model_params = ModelParams(parser)
    opt_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument('--gpu_id', type=str, default="0")  # 指定使用的 GPU
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if USE_NETWORK_GUI:
        # Start GUI server, configure and run training
        # [NOTE] 如果 GUI 服务器的 IP 和端口被占用，则无法同时使用两张显卡进行训练
        # 为避免不必要的麻烦，这里设置了 USE_NETWORK_GUI 为 False
        network_gui.init(args.ip, args.port)

    # 设置使用的 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(model_params.extract(args), opt_params.extract(args), pipeline_params.extract(args), args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
