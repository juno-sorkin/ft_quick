# pyright: reportMissingImports=false
import os
import sys
from typing import Iterable, List, Set, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.pytorch_utils import Conv1D as HFConv1D  # GPT2-style Linear
except Exception as import_error:
    print(f"Failed to import transformers: {import_error}", file=sys.stderr)
    raise

try:
    # Allows building the model on the 'meta' device without allocating real weights
    from accelerate import init_empty_weights
except Exception:
    # Fallback no-op context if accelerate is not available (should be installed per environment.yml)
    from contextlib import contextmanager

    @contextmanager
    def init_empty_weights():  # type: ignore
        yield


MODEL_ID = "unsloth/gpt-oss-20b"
OUTPUT_DIR = os.path.join("data_box", "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "unsloth_gpt_oss_20b_arch_and_modules.txt")


def ensure_output_dir_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_linear_like(module: nn.Module) -> bool:
    # Detect Linear-like modules commonly used for LoRA targeting
    linear_like_types: Tuple[type, ...] = (nn.Linear,)
    try:
        linear_like_types = (nn.Linear, HFConv1D)
    except Exception:
        pass
    return isinstance(module, linear_like_types)


def is_norm_or_embedding(name: str, module: nn.Module) -> bool:
    lowered = name.lower()
    if isinstance(module, (nn.LayerNorm, nn.Embedding)):
        return True
    if "norm" in lowered or "embed" in lowered or "lm_head" in lowered:
        return True
    return False


def collect_targetable_modules(model: nn.Module) -> Tuple[List[str], List[str]]:
    fully_qualified_names: List[str] = []
    leaf_module_names: Set[str] = set()

    for module_name, module in model.named_modules():
        if module_name == "":
            # Skip the root module itself
            continue

        if is_norm_or_embedding(module_name, module):
            continue

        if not is_linear_like(module):
            continue

        # Verify this module has at least one 2D parameter (typical for Linear)
        has_2d_weight = any(
            p.ndim == 2 for p in module.parameters(recurse=False)
        )
        if not has_2d_weight:
            continue

        fully_qualified_names.append(module_name)
        leaf_name = module_name.split(".")[-1]
        leaf_module_names.add(leaf_name)

    # Sort for deterministic output
    fully_qualified_names.sort()
    unique_leaf_names_sorted = sorted(leaf_module_names)
    return fully_qualified_names, unique_leaf_names_sorted


def build_model_from_config(model_id: str):
    # Load config only; do not download weights
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return config, model


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ensure_output_dir_exists(OUTPUT_DIR)

    print(f"Loading config and building empty model for: {MODEL_ID}")
    config, model = build_model_from_config(MODEL_ID)

    # Prepare architecture summary
    arch_lines: List[str] = []
    arch_lines.append(f"Model ID: {MODEL_ID}")
    arch_lines.append(f"Model class: {model.__class__.__name__}")
    arch_lines.append(f"Config class: {config.__class__.__name__}")
    arch_lines.append("")
    arch_lines.append("=== Configuration ===")
    arch_lines.append(str(config))
    arch_lines.append("")
    arch_lines.append("=== Architecture (empty weights) ===")
    arch_lines.append(str(model))

    # Collect targetable modules
    fq_names, leaf_names = collect_targetable_modules(model)

    report_lines: List[str] = []
    report_lines.extend(arch_lines)
    report_lines.append("")
    report_lines.append("=== Targetable module leaf names (unique) ===")
    for name in leaf_names:
        report_lines.append(name)

    report_lines.append("")
    report_lines.append("=== Targetable modules (fully-qualified) ===")
    for name in fq_names:
        report_lines.append(name)

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # Also print a concise stdout summary and the path
    print("\n".join(arch_lines[:6]))
    print("")
    print("First 20 targetable leaf names:")
    print(", ".join(leaf_names[:20]))
    print("")
    print(f"Wrote full architecture and module lists to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


