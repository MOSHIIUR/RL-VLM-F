import json
import os
import re
from functools import lru_cache
from importlib import metadata

import numpy as np
import torch
from PIL import Image


LABEL_PATTERN = re.compile(r"(?<!\d)(-1|0|1)(?!\d)")
_VLLM_DISABLED = False
_MIN_VLLM_VERSION_FOR_QWEN25 = (0, 7, 2)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_json(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return json.loads(value)


def _max_new_tokens(default=16):
    return _env_int("QWEN_VL_MAX_NEW_TOKENS", default)


def _query_text(goal_prompt):
    return (
        "Compare Image 1 and Image 2.\n"
        f"The goal is {goal_prompt}.\n"
        "Reply with only one line containing:\n"
        "0 if Image 1 better achieves the goal,\n"
        "1 if Image 2 better achieves the goal,\n"
        "-1 if they are equally good or if the answer is unclear."
    )


def _resize_image(image):
    max_side = _env_int("QWEN_VL_MAX_SIDE", 0)
    if max_side <= 0:
        return image
    width, height = image.size
    current_max = max(width, height)
    if current_max <= max_side:
        return image
    scale = max_side / float(current_max)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resample = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize(new_size, resample)


def _as_pil(image):
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")
    return _resize_image(pil_image)


def _resolve_dtype(dtype_name):
    if dtype_name == "auto":
        return "auto"
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(f"Unsupported QWEN_VL_DTYPE: {dtype_name}")


def _version_key(version_string):
    parts = [int(part) for part in re.findall(r"\d+", version_string)]
    if not parts:
        return (0,)
    return tuple(parts)


def _installed_vllm_version():
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return None


def _requested_model_name():
    return os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def _requires_qwen25_support():
    return "qwen2.5-vl" in _requested_model_name().lower()


def _vllm_support_status():
    version_string = _installed_vllm_version()
    if version_string is None:
        return False, "vllm_not_installed"
    if _requires_qwen25_support() and _version_key(version_string) < _MIN_VLLM_VERSION_FOR_QWEN25:
        minimum = ".".join(str(part) for part in _MIN_VLLM_VERSION_FOR_QWEN25)
        return False, f"vllm_{version_string}_too_old_for_qwen2.5-vl_need>={minimum}"
    return True, f"vllm_{version_string}"


def qwen_backend_summary():
    requested = os.environ.get("QWEN_VL_BACKEND", "auto").lower()
    if _VLLM_DISABLED:
        return f"transformers|requested={requested}|reason=vllm_disabled_after_failure"
    if requested == "transformers":
        return "transformers|requested=transformers"

    supported, reason = _vllm_support_status()
    if requested == "vllm":
        if supported:
            return f"vllm|requested=vllm|status={reason}"
        return f"transformers|requested=vllm|reason={reason}"

    if supported:
        return f"vllm|requested=auto|status={reason}"
    return f"transformers|requested=auto|reason={reason}"


def qwen_backend_name():
    return qwen_backend_summary().split("|", 1)[0]


@lru_cache(maxsize=1)
def _load_qwen_processor():
    from transformers import AutoProcessor

    model_name = _requested_model_name()
    processor_kwargs = {}
    min_pixels = os.environ.get("QWEN_VL_MIN_PIXELS")
    max_pixels = os.environ.get("QWEN_VL_MAX_PIXELS")
    if min_pixels:
        processor_kwargs["min_pixels"] = int(min_pixels)
    if max_pixels:
        processor_kwargs["max_pixels"] = int(max_pixels)

    return AutoProcessor.from_pretrained(model_name, **processor_kwargs)


@lru_cache(maxsize=1)
def _load_qwen_transformers():
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-VL requires a recent transformers install. "
            "Install a newer transformers build before using vlm=qwen_local."
        ) from exc

    model_name = _requested_model_name()
    load_kwargs = {
        "device_map": os.environ.get("QWEN_VL_DEVICE_MAP", "auto"),
        "torch_dtype": _resolve_dtype(os.environ.get("QWEN_VL_DTYPE", "auto")),
    }
    attn_impl = os.environ.get("QWEN_VL_ATTN_IMPLEMENTATION")
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    processor = _load_qwen_processor()
    return model, processor


@lru_cache(maxsize=1)
def _load_qwen_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "QWEN_VL_BACKEND=vllm was requested, but vLLM is not installed."
        ) from exc

    supported, reason = _vllm_support_status()
    if not supported:
        raise RuntimeError(f"Requested vLLM backend is unavailable: {reason}")

    model_name = _requested_model_name()
    llm_kwargs = {
        "model": model_name,
        "dtype": os.environ.get("QWEN_VL_DTYPE", "auto"),
        "gpu_memory_utilization": _env_float("QWEN_VL_GPU_MEMORY_UTILIZATION", 0.55),
        "max_model_len": _env_int("QWEN_VL_MAX_MODEL_LEN", 4096),
        "limit_mm_per_prompt": _env_json("QWEN_VL_LIMIT_MM_PER_PROMPT", {"image": 2}),
        "max_num_seqs": _env_int("QWEN_VL_MAX_NUM_SEQS", 4),
    }
    if os.environ.get("QWEN_VL_ENFORCE_EAGER"):
        llm_kwargs["enforce_eager"] = os.environ.get("QWEN_VL_ENFORCE_EAGER") == "1"
    quantization = os.environ.get("QWEN_VL_QUANTIZATION")
    if quantization:
        llm_kwargs["quantization"] = quantization

    llm = LLM(**llm_kwargs)
    return llm, SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=_max_new_tokens(),
    )


def _model_device(model):
    model_device = getattr(model, "device", None)
    if model_device is None or str(model_device) == "meta":
        model_device = next(model.parameters()).device
    return model_device


def _extract_label(text):
    matches = LABEL_PATTERN.findall(text)
    if not matches:
        return -1
    label = int(matches[-1])
    if label not in (-1, 0, 1):
        return -1
    return label


def _transformers_compare(image_1, image_2, goal_prompt, max_new_tokens):
    model, processor = _load_qwen_transformers()
    pil_images = [_as_pil(image_1), _as_pil(image_2)]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_images[0]},
                {"type": "image", "image": pil_images[1]},
                {"type": "text", "text": _query_text(goal_prompt)},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt],
        images=pil_images,
        padding=True,
        return_tensors="pt",
    )
    model_device = _model_device(model)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return _extract_label(output_text)


def _vllm_compare_batch(image_pairs, goal_prompt):
    llm, sampling_params = _load_qwen_vllm()
    processor = _load_qwen_processor()
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    requests = []
    for image_1, image_2 in image_pairs:
        pil_images = [_as_pil(image_1), _as_pil(image_2)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[0]},
                    {"type": "image", "image": pil_images[1]},
                    {"type": "text", "text": _query_text(goal_prompt)},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mm_data = {}
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
        else:
            mm_data["image"] = pil_images
        requests.append(
            {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
        )

    batch_size = _env_int("QWEN_VL_BATCH_SIZE", 4)
    labels = []
    for start in range(0, len(requests), batch_size):
        outputs = llm.generate(
            requests[start:start + batch_size],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for output in outputs:
            text = output.outputs[0].text.strip() if output.outputs else ""
            labels.append(_extract_label(text))
    return labels


def qwen_compare_batch(image_pairs, goal_prompt, max_new_tokens=None):
    global _VLLM_DISABLED
    if max_new_tokens is None:
        max_new_tokens = _max_new_tokens()

    backend = qwen_backend_name()
    if backend == "vllm":
        try:
            return _vllm_compare_batch(image_pairs, goal_prompt)
        except Exception as exc:
            _VLLM_DISABLED = True
            print(f"qwen_local: disabling vllm backend after failure: {exc}", flush=True)

    return [
        _transformers_compare(image_1, image_2, goal_prompt, max_new_tokens=max_new_tokens)
        for image_1, image_2 in image_pairs
    ]


def qwen_compare(image_1, image_2, goal_prompt, max_new_tokens=None):
    return qwen_compare_batch(
        [(image_1, image_2)],
        goal_prompt,
        max_new_tokens=max_new_tokens,
    )[0]
