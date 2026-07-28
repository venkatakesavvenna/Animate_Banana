# Academic / public benchmark datasets.
# TODO: organize into domain subfolders (vqa/, document/, science/, chart/)
#       in a follow-up once the full set is stable.
#
# HF-hosted (streamable): chartqa, textvqa, docvqa, vqa2, okvqa, aokvqa,
#   scienceqa, infovqa, mathvista, realworldqa, mmmu, ai2d, vsr
# Local-download (data_path required): dvqa, tallyqa, plotqa, figureqa,
#   tabwmp, scenetextqa, countbenchqa, clockbench, hateful_memes, vizwiz
import importlib
import logging

_logger = logging.getLogger(__name__)

_PUBLIC_MODULES = [
    # HF-hosted
    "chartqa",
    "textvqa",
    "docvqa",
    "vqa2",
    "okvqa",
    "aokvqa",
    "scienceqa",
    "infovqa",
    "mathvista",
    "realworldqa",
    "mmmu",
    "ai2d",
    "vsr",
    # Local-download
    "dvqa",
    "tallyqa",
    "plotqa",
    "figureqa",
    "tabwmp",
    "scenetextqa",
    "countbenchqa",
    "clockbench",
    "hateful_memes",
    "vizwiz",
]

for _name in _PUBLIC_MODULES:
    try:
        importlib.import_module(f"img_2_svg_pretraining.training.training_core.datasets.public.{_name}")
    except Exception as _e:
        _logger.debug("Skipping public dataset %s: %s", _name, _e)
