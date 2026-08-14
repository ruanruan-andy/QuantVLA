"""Train GAP-OPQD on LIBERO-Plus OOD calibration trajectories.

The simulator runs in a separate ``libero_test`` process, while this trainer
stays in ``groot_test``.  The simulator rollout is non-differentiable; frozen
student backbone features are cached and only the action-head LoRA is
re-forwarded with gradients.  A second clean LIBERO process supplies the IID
anchor used to limit clean-domain regression.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from tqdm.auto import tqdm
from transformers.feature_extraction_utils import BatchFeature
import zmq

from examples.Libero.custom_data_config import LiberoDataConfig, LiberoDataConfigMeanStd
from gr00t.experiment.gap_opqd import GAPOPQDConfig, build_gap_opqd_targets
from gr00t.model.policy import Gr00tPolicy, unsqueeze_dict_values


SUITE_MODEL_PATHS = {
    "libero_spatial": "youliangtan/gr00t-n1.5-libero-spatial-posttrain",
    "libero_goal": "youliangtan/gr00t-n1.5-libero-goal-posttrain",
    "libero_object": "youliangtan/gr00t-n1.5-libero-object-posttrain",
    "libero_10": "youliangtan/gr00t-n1.5-libero-long-posttrain",
}
SUITE_CALIBRATION_NAMES = {
    "libero_spatial": "spatial",
    "libero_goal": "goal",
    "libero_object": "object",
    "libero_10": "long",
}
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@dataclass
class TrainConfig:
    task_suite_name: str = "libero_spatial"
    output_dir: str = "output/gap-opqd-ood/libero_spatial"
    sample_manifest: str = "configs/libero_plus/first_24_per_category.json"
    max_iterations: int = 100
    rollout_horizon: int = 16
    num_steps_wait: int = 10
    task_ids: list[int] | None = None
    initial_state_ids: list[int] = field(default_factory=lambda: [0])
    seed: int = 20260813
    env_host: str = "127.0.0.1"
    env_port: int = 5590
    clean_env_host: str = "127.0.0.1"
    clean_env_port: int = 5591
    clean_task_ids: list[int] = field(default_factory=lambda: list(range(10)))
    clean_initial_state_ids: list[int] = field(default_factory=lambda: list(range(40)))
    clean_anchor_rollout_horizon: int = 4

    device: str = "cuda:0"
    denoising_steps: int = 8
    action_dims: int = 7

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0

    temporal_horizon: int = 4
    temporal_discount: float = 0.9
    alpha_q: float = 1.0
    beta_r: float = 1.0
    weight_min: float = 1.0
    weight_max: float = 5.0

    lambda_anchor: float = 0.1
    anchor_replay_size: int = 256
    anchor_batch_size: int = 4
    save_every: int = 100
    resume: bool = True
    resume_from_checkpoint: str | None = None
    dry_run: bool = False

    model_path: str | None = None
    calibration_path: str | None = None
    quant_pack_dir: str | None = None

    def validate(self) -> None:
        if self.task_suite_name not in SUITE_MODEL_PATHS:
            raise ValueError(f"unsupported suite: {self.task_suite_name}")
        if self.max_iterations <= 0 or self.rollout_horizon <= 0:
            raise ValueError("max_iterations and rollout_horizon must be positive")
        if self.task_ids is not None and not self.task_ids:
            raise ValueError("task_ids must be None or non-empty")
        if not self.initial_state_ids:
            raise ValueError("initial_state_ids must not be empty")
        if self.action_dims <= 0:
            raise ValueError("action_dims must be positive")
        if self.lambda_anchor < 0.0:
            raise ValueError("lambda_anchor must be non-negative")
        if self.lambda_anchor > 0.0 and (
            not self.clean_task_ids
            or not self.clean_initial_state_ids
            or self.clean_anchor_rollout_horizon <= 0
        ):
            raise ValueError("clean anchor sampling settings must be positive and non-empty")
        if self.anchor_replay_size < 0 or self.anchor_batch_size < 0:
            raise ValueError("anchor replay settings must be non-negative")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")


@dataclass
class CachedState:
    backbone_output: BatchFeature
    action_input: BatchFeature
    initial_noise: torch.Tensor
    teacher_action: torch.Tensor


class IIDEnvClient:
    """Small trusted-localhost RPC client for the clean LIBERO process."""

    def __init__(self, host: str, port: int, timeout_ms: int = 120_000):
        self.endpoint = f"tcp://{host}:{port}"
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(self.endpoint)

    def call(self, endpoint: str, **data):
        self.socket.send(pickle.dumps({"endpoint": endpoint, "data": data}))
        response = pickle.loads(self.socket.recv())
        if "error" in response:
            raise RuntimeError(f"LIBERO environment service: {response['error']}")
        return response["result"]

    def wait_until_ready(self, timeout_seconds: float = 60.0) -> None:
        deadline = time.time() + timeout_seconds
        self.socket.setsockopt(zmq.RCVTIMEO, 1_000)
        self.socket.setsockopt(zmq.SNDTIMEO, 1_000)
        while True:
            try:
                self.call("ping")
                self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                return
            except (zmq.ZMQError, RuntimeError):
                if time.time() >= deadline:
                    raise TimeoutError("clean LIBERO environment service did not become ready")
                self.socket.close(linger=0)
                self.socket = self.context.socket(zmq.REQ)
                self.socket.setsockopt(zmq.RCVTIMEO, 1_000)
                self.socket.setsockopt(zmq.SNDTIMEO, 1_000)
                self.socket.connect(self.endpoint)
                time.sleep(0.5)

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def _data_config_for_suite(suite_name: str):
    return LiberoDataConfigMeanStd() if suite_name == "libero_goal" else LiberoDataConfig()


def _clear_quantization_environment() -> None:
    prefixes = ("GR00T_DUQUANT_", "GR00T_ATM_", "GR00T_OHB_")
    for key in list(os.environ):
        if key.startswith(prefixes):
            del os.environ[key]


def _configure_quantvla(config: TrainConfig, repo_root: Path) -> None:
    calibration_name = SUITE_CALIBRATION_NAMES[config.task_suite_name]
    calibration_path = Path(
        config.calibration_path or repo_root / f"atm_alpha_beta_{calibration_name}.json"
    ).resolve()
    quant_pack_dir = Path(
        config.quant_pack_dir
        or repo_root / "model" / "quantvla" / "groot-n1.5" / config.task_suite_name / "duquant_pack"
    ).resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"QuantVLA calibration file is missing: {calibration_path}")
    quant_pack_dir.mkdir(parents=True, exist_ok=True)

    os.environ.update(
        {
            "GR00T_DUQUANT_DEBUG": "0",
            "GR00T_DUQUANT_SCOPE": "",
            "GR00T_DUQUANT_INCLUDE": (
                r".*(backbone\.eagle_model\.language_model\..*\."
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|"
                r"action_head\.model\.transformer_blocks\.\d+\.ff\.net\."
                r"(0\.proj|2)).*"
            ),
            "GR00T_DUQUANT_EXCLUDE": (
                r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)"
            ),
            "GR00T_DUQUANT_WBITS_DEFAULT": "4",
            "GR00T_DUQUANT_ABITS": "8",
            "GR00T_DUQUANT_BLOCK": "64",
            "GR00T_DUQUANT_PERMUTE": "0",
            "GR00T_DUQUANT_ROW_ROT": "restore",
            "GR00T_DUQUANT_ACT_PCT": "99.9",
            "GR00T_DUQUANT_CALIB_STEPS": "32",
            "GR00T_DUQUANT_LS": "0.15",
            "GR00T_DUQUANT_PACKDIR": str(quant_pack_dir),
            "GR00T_ATM_ALPHA_PATH": str(calibration_path),
            "GR00T_ATM_ENABLE": "1",
            "GR00T_ATM_SCOPE": "dit",
            "GR00T_OHB_ENABLE": "1",
            "GR00T_OHB_FALLBACK": "1.0",
            "GR00T_OHB_SCOPE": "dit",
        }
    )


def _load_policy(config: TrainConfig, *, quantized: bool, repo_root: Path) -> Gr00tPolicy:
    if quantized:
        _configure_quantvla(config, repo_root)
    else:
        _clear_quantization_environment()

    data_config = _data_config_for_suite(config.task_suite_name)
    policy = Gr00tPolicy(
        model_path=config.model_path or SUITE_MODEL_PATHS[config.task_suite_name],
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        denoising_steps=config.denoising_steps,
        device=config.device,
    )
    policy.model.eval()
    return policy


def _attach_action_head_lora(policy: Gr00tPolicy, config: TrainConfig) -> list[str]:
    target_modules = []
    for name, module in policy.model.named_modules():
        if "action_head" not in name or not isinstance(module, nn.Linear):
            continue
        if any(token in name for token in ("to_q", "to_k", "to_v")):
            target_modules.append(name)
    if not target_modules:
        raise RuntimeError("no action-head attention projections were found for LoRA")

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    policy.model = get_peft_model(policy.model, lora_config)
    policy.model.eval()
    policy.model.print_trainable_parameters()
    return target_modules


def _base_model(policy: Gr00tPolicy):
    get_base_model = getattr(policy.model, "get_base_model", None)
    return get_base_model() if get_base_model is not None else policy.model


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion.copy()
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quaternion[3] * quaternion[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return quaternion[:3] * 2.0 * math.acos(float(quaternion[3])) / denominator


def _get_libero_images(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        observation["agentview_image"][::-1, ::-1],
        observation["robot0_eye_in_hand_image"][::-1, ::-1],
    )


def _normalize_gripper_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[..., -1] = np.sign(1.0 - 2.0 * action[..., -1])
    return action


def _process_observation(observation: dict[str, Any], language: str) -> dict[str, Any]:
    xyz = observation["robot0_eef_pos"]
    rpy = _quat_to_axis_angle(observation["robot0_eef_quat"])
    gripper = observation["robot0_gripper_qpos"]
    image, wrist_image = _get_libero_images(observation)
    return {
        "video.image": np.expand_dims(image, axis=0),
        "video.wrist_image": np.expand_dims(wrist_image, axis=0),
        "state.x": np.array([[xyz[0]]]),
        "state.y": np.array([[xyz[1]]]),
        "state.z": np.array([[xyz[2]]]),
        "state.roll": np.array([[rpy[0]]]),
        "state.pitch": np.array([[rpy[1]]]),
        "state.yaw": np.array([[rpy[2]]]),
        "state.gripper": np.expand_dims(gripper, axis=0),
        "annotation.human.action.task_description": [language],
    }


def _normalize_observation(policy: Gr00tPolicy, observation: dict[str, Any]) -> dict[str, Any]:
    batched = unsqueeze_dict_values(observation)
    for key, value in batched.items():
        if not isinstance(value, np.ndarray):
            batched[key] = np.asarray(value)
    return policy.apply_transforms(batched)


def _copy_feature(feature: BatchFeature, device: str | torch.device) -> BatchFeature:
    copied = {}
    for key, value in feature.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"cached BatchFeature value {key!r} is not a tensor")
        copied[key] = value.detach().to(device=device)
    return BatchFeature(data=copied)


def _cache_feature(feature: BatchFeature) -> BatchFeature:
    return _copy_feature(feature, device="cpu")


def _prepare_backbone(model, normalized_input: dict[str, Any]):
    backbone_input, action_input = model.prepare_input(normalized_input)
    with torch.no_grad():
        backbone_output = model.backbone(backbone_input)
    required_action_input = BatchFeature(
        data={key: action_input[key] for key in ("state", "embodiment_id")}
    )
    return backbone_output, required_action_input


def _sample_from_cached(
    model,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    initial_noise: torch.Tensor,
    *,
    with_grad: bool,
) -> torch.Tensor:
    device = model.action_head.device
    backbone = _copy_feature(backbone_output, device=device)
    actions = _copy_feature(action_input, device=device)
    noise = initial_noise.to(device=device, dtype=model.action_head.dtype)
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model.action_head.sample_actions(
            backbone, actions, initial_noise=noise
        )["action_pred"]
    return prediction


def _normalized_to_libero_action(policy: Gr00tPolicy, normalized_action: torch.Tensor) -> np.ndarray:
    action_dict = policy._get_unnormalized_action(normalized_action.detach().float().cpu())
    components = []
    for key in ACTION_KEYS:
        value = np.asarray(action_dict[f"action.{key}"])
        if value.ndim < 2:
            raise ValueError(f"unexpected unnormalized action shape for {key}: {value.shape}")
        components.append(float(value.reshape(value.shape[0], value.shape[1], -1)[0, 0, 0]))
    action = np.asarray(components, dtype=np.float32)
    return _normalize_gripper_action(action)


def _rollout(
    config: TrainConfig,
    teacher: Gr00tPolicy,
    student: Gr00tPolicy,
    env_client: IIDEnvClient,
    *,
    task_id: int,
    initial_state_id: int,
    rollout_horizon: int,
    noise_generator: torch.Generator,
) -> tuple[list[CachedState], torch.Tensor, torch.Tensor, bool]:
    teacher_model = _base_model(teacher)
    student_model = _base_model(student)
    states: list[CachedState] = []
    teacher_actions = []
    student_actions = []
    success = False
    reset_result = env_client.call(
        "reset",
        suite=config.task_suite_name,
        task_id=task_id,
        initial_state_id=initial_state_id,
        num_steps_wait=config.num_steps_wait,
        horizon=config.num_steps_wait + rollout_horizon + 1,
    )
    observation = reset_result["observation"]
    language = reset_result["language"]
    if reset_result["done"]:
        return states, torch.empty(0), torch.empty(0), True

    for _ in range(rollout_horizon):
        policy_observation = _process_observation(observation, language)
        normalized_input = _normalize_observation(student, policy_observation)
        teacher_backbone, teacher_action_input = _prepare_backbone(
            teacher_model, normalized_input
        )
        student_backbone, student_action_input = _prepare_backbone(
            student_model, normalized_input
        )

        noise = torch.randn(
            1,
            student_model.action_head.config.action_horizon,
            student_model.action_head.config.action_dim,
            generator=noise_generator,
            dtype=torch.float32,
        )
        teacher_prediction = _sample_from_cached(
            teacher_model,
            teacher_backbone,
            teacher_action_input,
            noise,
            with_grad=False,
        )
        student_prediction = _sample_from_cached(
            student_model,
            student_backbone,
            student_action_input,
            noise,
            with_grad=False,
        )

        teacher_cpu = teacher_prediction.detach().float().cpu()
        student_cpu = student_prediction.detach().float().cpu()
        states.append(
            CachedState(
                backbone_output=_cache_feature(student_backbone),
                action_input=_cache_feature(student_action_input),
                initial_noise=noise.cpu(),
                teacher_action=teacher_cpu,
            )
        )
        teacher_actions.append(teacher_cpu.squeeze(0))
        student_actions.append(student_cpu.squeeze(0))

        libero_action = _normalized_to_libero_action(student, student_prediction)
        step_result = env_client.call("step", action=libero_action.tolist())
        observation = step_result["observation"]
        if step_result["done"]:
            success = True
            break

    if not states:
        return states, torch.empty(0), torch.empty(0), success
    return states, torch.stack(student_actions), torch.stack(teacher_actions), success


def _backward_state_loss(
    student_model,
    state: CachedState,
    *,
    action_dims: int,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    prediction = _sample_from_cached(
        student_model,
        state.backbone_output,
        state.action_input,
        state.initial_noise,
        with_grad=True,
    )
    target = state.teacher_action.to(device=prediction.device, dtype=prediction.dtype)
    loss = (prediction[:, 0, :action_dims] - target[:, 0, :action_dims]).float().square().mean()
    (loss * torch.as_tensor(scale, device=loss.device, dtype=loss.dtype)).backward()
    return loss.detach()


def _save_checkpoint(
    output_dir: Path,
    student: Gr00tPolicy,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    *,
    sampler: random.Random,
    noise_generator: torch.Generator,
    task_schedule: list[int],
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{iteration:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    student.model.save_pretrained(checkpoint_dir / "adapter")
    torch.save(
        {
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "sampler_state": sampler.getstate(),
            "noise_generator_state": noise_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "task_schedule": task_schedule,
        },
        checkpoint_dir / "trainer_state.pt",
    )


def _resolve_resume_checkpoint(config: TrainConfig, output_dir: Path) -> Path | None:
    if config.resume_from_checkpoint is not None:
        checkpoint = Path(config.resume_from_checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        return checkpoint
    if not config.resume:
        return None
    candidates = sorted(
        path
        for path in output_dir.glob("checkpoint-*")
        if (path / "adapter").is_dir() and (path / "trainer_state.pt").is_file()
    )
    return candidates[-1] if candidates else None


def _truncate_metrics_to_iteration(metrics_path: Path, iteration: int) -> None:
    if not metrics_path.exists():
        return
    retained = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["iteration"]) <= iteration:
            retained.append(record)
    metrics_path.write_text(
        "".join(json.dumps(record) + "\n" for record in retained), encoding="utf-8"
    )


def train(config: TrainConfig) -> None:
    config.validate()
    method_config = GAPOPQDConfig(
        temporal_horizon=config.temporal_horizon,
        temporal_discount=config.temporal_discount,
        alpha_q=config.alpha_q,
        beta_r=config.beta_r,
        weight_min=config.weight_min,
        weight_max=config.weight_max,
    )
    method_config.validate()
    if config.dry_run:
        print(json.dumps(dataclasses.asdict(config), indent=2))
        print(json.dumps(dataclasses.asdict(method_config), indent=2))
        return

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = _resolve_resume_checkpoint(config, output_dir)
    serialized_config = dataclasses.asdict(config)
    serialized_config["resolved_resume_checkpoint"] = (
        str(resume_checkpoint) if resume_checkpoint is not None else None
    )
    (output_dir / "config.json").write_text(
        json.dumps(serialized_config, indent=2), encoding="utf-8"
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    sampler = random.Random(config.seed)
    noise_generator = torch.Generator(device="cpu").manual_seed(config.seed)

    env_client = IIDEnvClient(config.env_host, config.env_port)
    clean_env_client = None
    try:
        env_client.wait_until_ready()
        selected_task_metadata = env_client.call("list_tasks")
        available_task_ids = {int(item["task_id"]) for item in selected_task_metadata}
        if config.task_ids is None:
            primary_task_ids = [int(item["task_id"]) for item in selected_task_metadata]
        else:
            invalid_task_ids = sorted(set(config.task_ids) - available_task_ids)
            if invalid_task_ids:
                raise ValueError(
                    f"task IDs are outside the OOD calibration manifest: {invalid_task_ids}"
                )
            primary_task_ids = list(dict.fromkeys(config.task_ids))
        category_by_task_id = {
            int(item["task_id"]): str(item["category"]) for item in selected_task_metadata
        }
        task_schedule = primary_task_ids.copy()
        sampler.shuffle(task_schedule)
        (output_dir / "selected_ood_tasks.json").write_text(
            json.dumps(
                [
                    item
                    for item in selected_task_metadata
                    if int(item["task_id"]) in set(primary_task_ids)
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"Loaded OOD calibration split: {len(primary_task_ids)} tasks from "
            f"{Path(config.sample_manifest).resolve()}"
        )

        if config.lambda_anchor > 0.0 and config.anchor_batch_size > 0:
            clean_env_client = IIDEnvClient(config.clean_env_host, config.clean_env_port)
            clean_env_client.wait_until_ready()

        print("Loading frozen full-precision teacher")
        teacher = _load_policy(config, quantized=False, repo_root=repo_root)
        teacher.model.requires_grad_(False)
        teacher.model.eval()

        print("Loading W4A8 QuantVLA student")
        student = _load_policy(config, quantized=True, repo_root=repo_root)
        if resume_checkpoint is None:
            lora_targets = _attach_action_head_lora(student, config)
            print(f"Attached LoRA to {len(lora_targets)} action-head projections")
        else:
            student.model = PeftModel.from_pretrained(
                student.model, resume_checkpoint / "adapter", is_trainable=True
            )
            student.model.eval()
            student.model.print_trainable_parameters()
            print(f"Loaded trainable LoRA adapter from {resume_checkpoint / 'adapter'}")
        trainable_parameters = [
            parameter for parameter in student.model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("student has no trainable LoRA parameters")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        start_iteration = 1
        if resume_checkpoint is not None:
            resume_state = torch.load(
                resume_checkpoint / "trainer_state.pt",
                map_location=config.device,
                weights_only=False,
            )
            optimizer.load_state_dict(resume_state["optimizer"])
            completed_iteration = int(resume_state["iteration"])
            start_iteration = completed_iteration + 1
            if "sampler_state" in resume_state:
                sampler.setstate(resume_state["sampler_state"])
            if "noise_generator_state" in resume_state:
                noise_generator.set_state(resume_state["noise_generator_state"].cpu())
            if "torch_rng_state" in resume_state:
                torch.set_rng_state(resume_state["torch_rng_state"].cpu())
            if "numpy_rng_state" in resume_state:
                np.random.set_state(resume_state["numpy_rng_state"])
            if "task_schedule" in resume_state:
                restored_schedule = [int(value) for value in resume_state["task_schedule"]]
                if set(restored_schedule) != set(primary_task_ids):
                    raise ValueError("resume checkpoint task schedule differs from manifest")
                task_schedule = restored_schedule
            print(f"Resuming training at iteration {start_iteration} from {resume_checkpoint}")

        student_model = _base_model(student)
        clean_replay: deque[CachedState] = deque(maxlen=config.anchor_replay_size)
        metrics_path = output_dir / "metrics.jsonl"
        if resume_checkpoint is not None:
            _truncate_metrics_to_iteration(metrics_path, start_iteration - 1)
        with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics_file:
            progress = tqdm(
                range(start_iteration, config.max_iterations + 1),
                initial=start_iteration - 1,
                total=config.max_iterations,
                desc=config.task_suite_name,
                dynamic_ncols=True,
            )
            for iteration in progress:
                started_at = time.time()
                schedule_index = (iteration - 1) % len(task_schedule)
                if schedule_index == 0 and iteration > 1:
                    sampler.shuffle(task_schedule)
                task_id = task_schedule[schedule_index]
                initial_state_id = sampler.choice(config.initial_state_ids)
                states, student_actions, teacher_actions, success = _rollout(
                    config,
                    teacher,
                    student,
                    env_client,
                    task_id=task_id,
                    initial_state_id=initial_state_id,
                    rollout_horizon=config.rollout_horizon,
                    noise_generator=noise_generator,
                )
                if not states:
                    print(f"iteration {iteration}: empty rollout; skipping update")
                    continue

                q_values, r_values, weights = build_gap_opqd_targets(
                    student_actions,
                    teacher_actions,
                    method_config,
                    action_dims=config.action_dims,
                )
                optimizer.zero_grad(set_to_none=True)
                main_losses = []
                normalized_weights = weights / weights.sum().clamp_min(1e-8)
                for state, scale in zip(states, normalized_weights, strict=True):
                    main_losses.append(
                        _backward_state_loss(
                            student_model,
                            state,
                            action_dims=config.action_dims,
                            scale=scale,
                        ).cpu()
                    )

                clean_anchor_steps = 0
                clean_anchor_success = False
                if clean_env_client is not None:
                    clean_states, _, _, clean_anchor_success = _rollout(
                        config,
                        teacher,
                        student,
                        clean_env_client,
                        task_id=sampler.choice(config.clean_task_ids),
                        initial_state_id=sampler.choice(config.clean_initial_state_ids),
                        rollout_horizon=config.clean_anchor_rollout_horizon,
                        noise_generator=noise_generator,
                    )
                    clean_anchor_steps = len(clean_states)
                    clean_replay.extend(clean_states)

                anchor_losses = []
                if (
                    config.lambda_anchor > 0.0
                    and config.anchor_batch_size > 0
                    and clean_replay
                ):
                    anchor_count = min(config.anchor_batch_size, len(clean_replay))
                    anchor_states = sampler.sample(list(clean_replay), anchor_count)
                    anchor_scale = config.lambda_anchor / anchor_count
                    for state in anchor_states:
                        anchor_losses.append(
                            _backward_state_loss(
                                student_model,
                                state,
                                action_dims=config.action_dims,
                                scale=anchor_scale,
                            ).cpu()
                        )

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip_norm
                )
                if not torch.isfinite(torch.as_tensor(gradient_norm)):
                    raise FloatingPointError(
                        f"non-finite gradient norm at iteration {iteration}"
                    )
                optimizer.step()

                record = {
                    "iteration": iteration,
                    "suite": config.task_suite_name,
                    "domain": "libero_plus_ood_calibration",
                    "task_id": task_id,
                    "category": category_by_task_id[task_id],
                    "ood_epoch": (iteration - 1) // len(task_schedule) + 1,
                    "initial_state_id": initial_state_id,
                    "rollout_steps": len(states),
                    "rollout_success": success,
                    "q_mean": q_values.mean().item(),
                    "q_max": q_values.max().item(),
                    "r_mean": r_values.mean().item(),
                    "r_max": r_values.max().item(),
                    "weight_mean": weights.mean().item(),
                    "weight_max": weights.max().item(),
                    "loss_opqd_unweighted_mean": torch.stack(main_losses).mean().item(),
                    "loss_anchor_mean": (
                        torch.stack(anchor_losses).mean().item() if anchor_losses else 0.0
                    ),
                    "clean_anchor_steps": clean_anchor_steps,
                    "clean_anchor_success": clean_anchor_success,
                    "gradient_norm": float(gradient_norm),
                    "clean_replay_size": len(clean_replay),
                    "duration_seconds": round(time.time() - started_at, 3),
                    "q_by_timestep": q_values.tolist(),
                    "r_by_timestep": r_values.tolist(),
                }
                metrics_file.write(json.dumps(record) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in record.items()
                            if not key.endswith("timestep")
                        }
                    )
                )
                progress.set_postfix(
                    category=record["category"],
                    q=f"{record['q_mean']:.3e}",
                    grad=f"{record['gradient_norm']:.3e}",
                )

                if iteration % config.save_every == 0 or iteration == config.max_iterations:
                    _save_checkpoint(
                        output_dir,
                        student,
                        optimizer,
                        iteration,
                        sampler=sampler,
                        noise_generator=noise_generator,
                        task_schedule=task_schedule,
                    )
    finally:
        env_client.close()
        if clean_env_client is not None:
            clean_env_client.close()


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
