"""Human-in-the-loop node annotation tool (Streamlit + SAM3).

Produces, for scientific-diagram nodes, a human-confirmed point plus a
human-confirmed pixel mask per node instance. This supervision feeds two
fine-tuning efforts: Molmo (node pointing) and SAM3 (node masking).

SAM3 is used exclusively through its point/box *interactive* interface
(`Sam3TrackerModel`, the SAM2-compatible visual-prompting mode) -- never the
text/concept-prompted `Sam3Model`. "Node" is a structural role, not a visual
category, so open-vocabulary text prompting does not work for it; a human
confirms every point and every mask instead.

Modules:
- datamodel: dataclasses, the instance state machine, RLE mask helpers.
- store:     one-JSON-per-image persistence with file locking.
- sam3_backend: embed-once-per-image / segment-per-click SAM3 wrapper.
- compositor: PIL rendering of base image + mask + point layers.
- app:       the Streamlit review app (one canvas, points + masks together).
- ingest:    offline Molmo proposal script (NOT part of the app).
"""
