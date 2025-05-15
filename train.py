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
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import torch
import torchvision
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
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, logging_intervals, testing_iterations, saving_iterations, checkpoint_iterations, vis_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # iter_after_densify = 0
    for iteration in range(first_iter, opt.iterations + 1):

        if not network_gui.disabled:
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

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, splitting_mats, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["splitting_matrices"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), logging_intervals, testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter: # or gaussians.get_xyz.shape[0] < 6000000:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, splitting_mats, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:

                    def visualize_densify_hook(masks):
                        import numpy as np
                        import copy
                        import cv2
                        from utils.pose_utils import generate_ellipse_path, generate_spiral_path
                        from utils.graphics_utils import getWorld2View2

                        render_path = os.path.join(scene.model_path, 'vis_densify', f'{iteration:05d}')
                        os.makedirs(render_path, exist_ok=True)

                        # training cams
                        # cams = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(0, len(scene.getTrainCameras()), 16)]

                        # render video
                        views = scene.getTestCameras()
                        view = copy.deepcopy(views[0])
                        render_poses = generate_ellipse_path(views)

                        
                        for k in masks.keys():
                            msk = masks[k]
                            color = torch.zeros(gaussians.get_xyz.shape[0], 3).to('cuda')
                            color[msk.squeeze(-1)] = 1.0 

                            # number of filtered Gaussians
                            num = msk.sum().long()


                            ## render video
                            size = (view.original_image.shape[2], view.original_image.shape[1])
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            final_video = cv2.VideoWriter(os.path.join(render_path, f'render_video_splitmap_{k}_{num}.mp4'), fourcc, 60, size)
                            for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
                                view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
                                view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
                                view.camera_center = view.world_view_transform.inverse()[3, :3]
                                rendering = render(view, gaussians, pipe, bg, override_color=color)

                                img = torch.clamp(rendering["render"], min=0., max=1.)
                                torchvision.utils.save_image(img, os.path.join(render_path, f'{idx:05d}_{k}_{num}' + ".png"))
                                video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
                                final_video.write(video_img)
                            final_video.release()


                        ## render rgb video
                        size = (view.original_image.shape[2], view.original_image.shape[1])
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        final_video = cv2.VideoWriter(os.path.join(render_path, f'render_video_rgb.mp4'), fourcc, 60, size)
                        for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
                            view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
                            view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
                            view.camera_center = view.world_view_transform.inverse()[3, :3]
                            render_pkg = render(view, gaussians, pipe, bg)
                            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                            torchvision.utils.save_image(image, os.path.join(render_path, f'{idx:05d}_rgb.png'))
                            video_img = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
                            final_video.write(video_img)

                    if iteration in vis_iterations:
                        gaussians.visualize_densify_hook = visualize_densify_hook
                    else:
                        gaussians.visualize_densify_hook = None

                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_info = gaussians.densify_and_prune(opt.densify_strategy, opt.densify_grad_threshold, opt.densify_S_threshold, 0.005, scene.cameras_extent, size_threshold)
                    write_dict_log(dataset, {'iteration': iteration, **densify_info})

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "train_log.txt"), 'w') as log_f:
        # create & clear the log file
        pass

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def write_dict_log(args, dict_msg, prefix=None):
    with open(os.path.join(args.model_path, "train_log.txt"), 'a') as log_f:
        if prefix is not None:
            log_f.write("{}: ".format(prefix))

        log_text = ", ".join(["{} = {}".format(key, value) for key, value in dict_msg.items()])
        log_f.write(log_text + "\n")

def training_report(args, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, logging_intervals, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    
    if iteration % logging_intervals == 0:
        write_dict_log(args, {
            'iteration': iteration,
            'total_points': scene.gaussians.get_xyz.shape[0],
            'loss': loss.item(),
            'l1_loss': Ll1.item(),
            'elapsed': elapsed,
            'memory': torch.cuda.memory_reserved()/1024**3
        })

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Num. Pts. {}".format(iteration, config['name'], l1_test, psnr_test, scene.gaussians.get_xyz.shape[0]))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                write_dict_log(args, {'iteration': iteration, 'l1_loss': l1_test, 'psnr': psnr_test}, prefix="[Eval {}]".format(config['name']))
    
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--no_gui', action='store_true', default=False)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--logging_intervals", type=int, default=100)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--vis_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.disabled = args.no_gui
    if not network_gui.disabled:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.logging_intervals, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.vis_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
