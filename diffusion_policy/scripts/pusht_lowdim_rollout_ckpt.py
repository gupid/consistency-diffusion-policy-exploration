if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import os
import pathlib
from typing import Dict, List, Tuple

import av
import dill
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)


def load_policy_from_checkpoint(ckpt_path: str, device: torch.device):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    workspace_target = cfg._target_
    module_name, class_name = workspace_target.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    workspace_cls = getattr(module, class_name)
    workspace = workspace_cls.create_from_checkpoint(ckpt_path)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    policy.reset()
    return policy, cfg


def set_torch_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_env_kind(cfg) -> str:
    runner_target = cfg.task.env_runner._target_
    if runner_target.endswith("PushTKeypointsRunner"):
        return "pusht"
    if runner_target.endswith("RobomimicLowdimRunner"):
        return "robomimic_lowdim"
    raise NotImplementedError(f"Unsupported env runner target: {runner_target}")


def make_pusht_env(cfg, output_path: str):
    runner_cfg = cfg.task.env_runner
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

    env = MultiStepWrapper(
        VideoRecordingWrapper(
            PushTKeypointsEnv(
                legacy=runner_cfg.legacy_test,
                keypoint_visible_rate=runner_cfg.keypoint_visible_rate,
                agent_keypoints=runner_cfg.agent_keypoints,
                **kp_kwargs,
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=runner_cfg.fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=getattr(runner_cfg, "crf", 22),
                thread_type="FRAME",
                thread_count=1,
            ),
            file_path=output_path,
        ),
        n_obs_steps=cfg.n_obs_steps + cfg.n_latency_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=runner_cfg.max_steps,
    )
    return env


def make_robomimic_lowdim_env(cfg, output_path: str):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    task_cfg = cfg.task
    runner_cfg = cfg.task.env_runner
    dataset_path = os.path.expanduser(task_cfg.dataset_path)
    obs_keys = list(task_cfg.obs_keys)
    render_hw = tuple(getattr(runner_cfg, "render_hw", (256, 256)))
    render_camera_name = getattr(runner_cfg, "render_camera_name", "agentview")
    fps = getattr(runner_cfg, "fps", 10)
    crf = getattr(runner_cfg, "crf", 22)
    steps_per_render = max(20 // fps, 1)

    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    if bool(getattr(task_cfg, "abs_action", False)):
        env_meta = dict(env_meta)
        env_meta["env_kwargs"] = dict(env_meta["env_kwargs"])
        env_meta["env_kwargs"]["controller_configs"] = dict(env_meta["env_kwargs"]["controller_configs"])
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )

    env = MultiStepWrapper(
        VideoRecordingWrapper(
            RobomimicLowdimWrapper(
                env=robomimic_env,
                obs_keys=obs_keys,
                init_state=None,
                render_hw=render_hw,
                render_camera_name=render_camera_name,
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=crf,
                thread_type="FRAME",
                thread_count=1,
            ),
            file_path=output_path,
            steps_per_render=steps_per_render,
        ),
        n_obs_steps=cfg.n_obs_steps + cfg.n_latency_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=runner_cfg.max_steps,
    )
    return env


def make_env(cfg, output_path: str):
    env_kind = infer_env_kind(cfg)
    if env_kind == "pusht":
        return make_pusht_env(cfg, output_path=output_path)
    if env_kind == "robomimic_lowdim":
        return make_robomimic_lowdim_env(cfg, output_path=output_path)
    raise AssertionError(f"Unhandled env kind: {env_kind}")


def build_obs_dict(env_kind: str, obs, cfg, past_action):
    n_obs_steps = cfg.n_obs_steps

    if env_kind == "pusht":
        obs_dim = obs.shape[-1] // 2
        np_obs_dict = {
            "obs": obs[None, :n_obs_steps, :obs_dim].astype(np.float32),
            "obs_mask": obs[None, :n_obs_steps, obs_dim:] > 0.5,
        }
    elif env_kind == "robomimic_lowdim":
        np_obs_dict = {
            "obs": obs[None, :n_obs_steps].astype(np.float32),
        }
    else:
        raise AssertionError(f"Unhandled env kind: {env_kind}")

    if bool(getattr(cfg, "past_action_visible", False)) and (past_action is not None):
        np_obs_dict["past_action"] = past_action[:, -(n_obs_steps - 1):].astype(np.float32)
    return np_obs_dict


def rollout_once(policy, cfg, seed: int, output_path: str):
    device = policy.device
    env_kind = infer_env_kind(cfg)
    output_path = str(pathlib.Path(output_path).absolute())
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, output_path=output_path)
    env.seed(seed)
    obs = env.reset()
    policy.reset()
    done = False
    past_action = None
    max_reward = float("-inf")

    while not done:
        np_obs_dict = build_obs_dict(
            env_kind=env_kind,
            obs=obs,
            cfg=cfg,
            past_action=past_action,
        )
        obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
        action = np_action_dict["action"][0, cfg.n_latency_steps:]

        obs, reward, done, info = env.step(action)
        done = bool(done)
        past_action = action[None]
        max_reward = max(max_reward, float(reward))

    video_path = env.render()
    env.close()
    return video_path, max_reward


def read_video_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 10.0
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def resize_frame_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    img = Image.fromarray(frame)
    target_width = round(frame.shape[1] * target_height / frame.shape[0])
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.asarray(img)


def add_label_bar(frame: np.ndarray, left_label: str, right_label: str, split_x: int) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bar_height = 24
    canvas = Image.new("RGB", (img.width, img.height + bar_height), (255, 255, 255))
    canvas.paste(img, (0, bar_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), left_label, fill=(15, 23, 42), font=font)
    draw.text((split_x + 8, 6), right_label, fill=(15, 23, 42), font=font)
    return np.asarray(canvas)


def merge_videos_to_gif(
        left_video_path: str,
        right_video_path: str,
        gif_path: str,
        left_label: str,
        right_label: str):
    import imageio.v2 as imageio

    left_frames, left_fps = read_video_frames(left_video_path)
    right_frames, right_fps = read_video_frames(right_video_path)
    target_height = min(left_frames[0].shape[0], right_frames[0].shape[0])
    left_frames = [resize_frame_to_height(frame, target_height) for frame in left_frames]
    right_frames = [resize_frame_to_height(frame, target_height) for frame in right_frames]

    frame_count = max(len(left_frames), len(right_frames))
    separator = np.full((target_height, 4, 3), 255, dtype=np.uint8)
    combined_frames = []
    for frame_idx in range(frame_count):
        left_frame = left_frames[min(frame_idx, len(left_frames) - 1)]
        right_frame = right_frames[min(frame_idx, len(right_frames) - 1)]
        combined = np.concatenate([left_frame, separator, right_frame], axis=1)
        combined = add_label_bar(
            frame=combined,
            left_label=left_label,
            right_label=right_label,
            split_x=left_frame.shape[1] + separator.shape[1],
        )
        combined_frames.append(combined)

    fps = min(left_fps, right_fps)
    gif_path = pathlib.Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, combined_frames, fps=fps, loop=0)
    return str(gif_path)


def run_single_rollout(checkpoint: str, seed: int, sample_seed: int, output: str, device: torch.device):
    set_torch_seed(sample_seed)
    policy, cfg = load_policy_from_checkpoint(checkpoint, device=device)
    video_path, max_reward = rollout_once(
        policy=policy,
        cfg=cfg,
        seed=seed,
        output_path=output,
    )
    inference_steps = getattr(policy, "num_inference_steps", None)
    del policy
    return {
        "video_path": video_path,
        "max_reward": max_reward,
        "inference_steps": inference_steps,
        "cfg": cfg,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run lowdim checkpoint rollouts for Push-T or Robomimic tasks and optionally export a comparison GIF."
    )

    single_group = parser.add_argument_group("single rollout")
    single_group.add_argument("--checkpoint", help="Path to checkpoint file for single-rollout mode.")
    single_group.add_argument("--output", help="Output mp4 path for single-rollout mode.")

    compare_group = parser.add_argument_group("compare rollout")
    compare_group.add_argument("--baseline-checkpoint", help="Left-side checkpoint for compare mode.")
    compare_group.add_argument("--agent-checkpoint", help="Right-side checkpoint for compare mode.")
    compare_group.add_argument(
        "--compare-dir",
        help="Output directory for compare mode. Writes baseline.mp4, agent.mp4, and comparison.gif into it.",
    )
    compare_group.add_argument("--baseline-label", default="Baseline", help="Left-side label in the GIF.")
    compare_group.add_argument("--agent-label", default="Agent", help="Right-side label in the GIF.")

    parser.add_argument("--seed", required=True, type=int, help="Environment seed to replay.")
    parser.add_argument(
        "--sample-seed",
        default=0,
        type=int,
        help="Torch random seed for policy sampling noise.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    compare_mode = bool(args.baseline_checkpoint or args.agent_checkpoint or args.compare_dir)
    if compare_mode:
        if not (args.baseline_checkpoint and args.agent_checkpoint and args.compare_dir):
            raise ValueError("--baseline-checkpoint, --agent-checkpoint, and --compare-dir are all required in compare mode.")

        compare_dir = pathlib.Path(args.compare_dir)
        baseline_output = compare_dir / f"baseline_seed{args.seed}.mp4"
        agent_output = compare_dir / f"agent_seed{args.seed}.mp4"
        gif_output = compare_dir / f"comparison_seed{args.seed}.gif"

        baseline_info = run_single_rollout(
            checkpoint=args.baseline_checkpoint,
            seed=args.seed,
            sample_seed=args.sample_seed,
            output=str(baseline_output),
            device=device,
        )
        agent_info = run_single_rollout(
            checkpoint=args.agent_checkpoint,
            seed=args.seed,
            sample_seed=args.sample_seed,
            output=str(agent_output),
            device=device,
        )
        gif_path = merge_videos_to_gif(
            left_video_path=baseline_info["video_path"],
            right_video_path=agent_info["video_path"],
            gif_path=str(gif_output),
            left_label=args.baseline_label,
            right_label=args.agent_label,
        )

        print(f"Baseline video: {baseline_info['video_path']}")
        print(f"Agent video: {agent_info['video_path']}")
        print(f"Comparison GIF: {gif_path}")
        print(f"Seed: {args.seed}")
        print(f"Sample seed: {args.sample_seed}")
        print(f"Baseline max reward: {baseline_info['max_reward']:.6f}")
        print(f"Agent max reward: {agent_info['max_reward']:.6f}")
        if baseline_info["inference_steps"] is not None:
            print(f"Baseline inference steps: {baseline_info['inference_steps']}")
        if agent_info["inference_steps"] is not None:
            print(f"Agent inference steps: {agent_info['inference_steps']}")
        return

    if not (args.checkpoint and args.output):
        raise ValueError("--checkpoint and --output are required in single-rollout mode.")

    result = run_single_rollout(
        checkpoint=args.checkpoint,
        seed=args.seed,
        sample_seed=args.sample_seed,
        output=args.output,
        device=device,
    )
    print(f"Saved video to: {result['video_path']}")
    print(f"Seed: {args.seed}")
    print(f"Sample seed: {args.sample_seed}")
    print(f"Max reward: {result['max_reward']:.6f}")
    if result["inference_steps"] is not None:
        print(f"Inference steps: {result['inference_steps']}")


if __name__ == "__main__":
    main()
