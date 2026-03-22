if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import pathlib

import dill
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
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


def make_env(cfg, output_path: str):
    runner_cfg = cfg.task.env_runner
    kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

    env = MultiStepWrapper(
        VideoRecordingWrapper(
            PushTKeypointsEnv(
                legacy=runner_cfg.legacy_test,
                keypoint_visible_rate=runner_cfg.keypoint_visible_rate,
                agent_keypoints=runner_cfg.agent_keypoints,
                **kp_kwargs
            ),
            video_recoder=VideoRecorder.create_h264(
                fps=runner_cfg.fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=22,
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


def rollout_once(policy, cfg, seed: int, output_path: str):
    device = policy.device
    output_path = str(pathlib.Path(output_path).absolute())
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    env = make_env(cfg, output_path=output_path)
    env.seed(seed)
    obs = env.reset()
    done = False
    past_action = None
    max_reward = float("-inf")

    while not done:
        do = obs.shape[-1] // 2
        np_obs_dict = {
            "obs": obs[None, :cfg.n_obs_steps, :do].astype("float32"),
            "obs_mask": obs[None, :cfg.n_obs_steps, do:] > 0.5,
        }
        if cfg.past_action_visible and (past_action is not None):
            np_obs_dict["past_action"] = past_action[:, -(cfg.n_obs_steps - 1):].astype("float32")

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one PushT lowdim rollout from a checkpoint and save the video."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--seed", required=True, type=int, help="Environment seed to replay.")
    parser.add_argument(
        "--sample-seed",
        default=0,
        type=int,
        help="Torch random seed for policy sampling noise.",
    )
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.sample_seed)
    device = torch.device(args.device)
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, device=device)
    video_path, max_reward = rollout_once(
        policy=policy,
        cfg=cfg,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Saved video to: {video_path}")
    print(f"Seed: {args.seed}")
    print(f"Sample seed: {args.sample_seed}")
    print(f"Max reward: {max_reward:.6f}")
    print(f"Inference steps: {policy.num_inference_steps}")


if __name__ == "__main__":
    main()
