"""Fine-tuning techniques written by the project team (ported from the CS781 capstone).

Files are copied unchanged except for lines marked `# FIX (Claude)`.

Contract (same as the capstone): apply(backbone, **cfg) -> None. Called once after the
pretrained weights are loaded; on entry every parameter has requires_grad=True, on exit
requires_grad must be True exactly on what should train. `backbone` is
decipher.llm.Backbone (.lm = the DecoderLM, .blocks, .kind = "decoder", .d_model).
kd.py instead provides kd_loss(student_logits, teacher_logits, tags, **cfg).
"""
import importlib

AVAILABLE = ["full_ft", "lora", "lorma", "rslora", "bitfit", "partial_ft", "ia3", "adapters"]


def get(name):
    if name not in AVAILABLE:
        raise ValueError(f"unknown technique {name!r}; available: {AVAILABLE}")
    return importlib.import_module(f"{__name__}.{name}")
