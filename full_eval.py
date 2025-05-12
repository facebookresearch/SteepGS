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
#

import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_video", action="store_true")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--scene_postfix", default="", required=False)
parser.add_argument("--train_args", default="")

parser.add_argument('--mipnerf360_outdoor', "-m360out", required=False, default=None, type=str)
parser.add_argument('--mipnerf360_indoor', "-m360in", required=False, default=None, type=str)
parser.add_argument("--tanksandtemples", "-tat", required=False, default=None, type=str)
parser.add_argument("--deepblending", "-db", required=False, default=None, type=str)

args, _ = parser.parse_known_args()

if not args.mipnerf360_outdoor:
    mipnerf360_outdoor_scenes = []


if not args.mipnerf360_indoor:
    mipnerf360_indoor_scenes = []


if not args.tanksandtemples:
    tanks_and_temples_scenes = []


if not args.deepblending:
    deep_blending_scenes = []

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

print('Scenes to be evaluated: ', all_scenes)

if not args.skip_training:
    common_args = f" --quiet --eval --test_iterations -1 --no_gui {args.train_args} "
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360_outdoor + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + args.scene_postfix + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360_indoor + "/" + scene
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + args.scene_postfix + common_args)
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + args.scene_postfix + common_args)
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + args.scene_postfix + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360_outdoor + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360_indoor + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + args.scene_postfix + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + args.scene_postfix + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + args.scene_postfix + "\" "

    os.system("python metrics.py -m " + scenes_string)


if not args.skip_video:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360_outdoor + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360_indoor + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --video --skip_train --skip_test --skip_save"
    for scene, source in zip(all_scenes, all_sources):
        if scene in mipnerf360_indoor_scenes or scene in mipnerf360_outdoor_scenes or scene in tanks_and_temples_scenes or scene in deep_blending_scenes:
            camera_path_args = ' --camera_path 360'
        else:
            camera_path_args = ''
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + args.scene_postfix + common_args + camera_path_args)
