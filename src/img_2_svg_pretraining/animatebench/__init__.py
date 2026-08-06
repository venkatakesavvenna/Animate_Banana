"""AnimateBench: quantitative evaluation of the AnimateBanana pipeline.

Scores a pipeline run's artifacts (diagram code, structure XML, animation
sequence, animation code) against the benchmark's reference bundle, using the
metric subset selected from the AnimateBench design document: programmatic
checks wherever the property is mechanical, Gemini as judge where it is not.

Entry point: python -m img_2_svg_pretraining.animatebench.run_eval
"""
