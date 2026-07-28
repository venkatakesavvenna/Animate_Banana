# Layout datasets — register themselves into DatasetRegistry on import.
# Import individual modules or this package to trigger registration.
# Modules that require local data files (hierlay, hierlay_v2) are imported
# with a try/except so that other datasets remain available when those
# local files are absent.
import importlib
import logging

_logger = logging.getLogger(__name__)

_ALWAYS_AVAILABLE = [
    "annopage",
    "comp_hr",
    "custom_prompt",
    "d4la",
    "doc_bank",
    "doclaynet",
    "docsyn",
    "indicdlp",
    "indicdlp_hier",
    "m6doc",
    "omni_docs",
    "prima",
    "promptable_v1",
    "publaynet",
]

# Datasets that open local files at import time — skip gracefully when absent.
_LOCAL_FILE_DEPENDENT = [
    "hierlay",
    "hierlay_v2",
]

for _name in _ALWAYS_AVAILABLE:
    importlib.import_module(f"img_2_svg_pretraining.training.training_core.datasets.layout.{_name}")

for _name in _LOCAL_FILE_DEPENDENT:
    try:
        importlib.import_module(f"img_2_svg_pretraining.training.training_core.datasets.layout.{_name}")
    except (FileNotFoundError, OSError) as _e:
        _logger.debug("Skipping layout dataset %s: %s", _name, _e)
