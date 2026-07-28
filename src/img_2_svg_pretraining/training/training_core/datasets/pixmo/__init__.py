import importlib
import logging

_logger = logging.getLogger(__name__)

_PIXMO_MODULES = [
    "pixmo_cap",
    "pixmo_cap_local",
    "pixmo_cap_qa",
    "pixmo_ask_model_anything",
    "pixmo_points",
    "pixmo_count",
    "pixmo_point_explanations",
    "pixmo_docs",
]

for _name in _PIXMO_MODULES:
    try:
        importlib.import_module(f"img_2_svg_pretraining.training.training_core.datasets.pixmo.{_name}")
    except Exception as _e:
        _logger.debug("Skipping pixmo dataset %s: %s", _name, _e)
