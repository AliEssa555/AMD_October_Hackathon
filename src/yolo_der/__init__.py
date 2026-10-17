"""YOLO-DER: Dynamic Enhancement Routing for adverse-weather / low-light detection.

Implements the DER and EnhanceNet modules of Gao, Omar & Mahmuddin, *Electronics* 2025,
14, 4851 (doi:10.3390/electronics14244851) on top of Ultralytics YOLOv8n, trained on the
ExDark low-light dataset on AMD Instinct MI300X.

Deviations from the paper are documented in ``docs/PROJECT_SPEC.md`` Section 2 and the README.
"""

__version__ = "0.1.0"
