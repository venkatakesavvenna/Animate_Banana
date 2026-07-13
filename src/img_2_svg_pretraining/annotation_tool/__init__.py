"""Human-in-the-loop raster-region annotation tool (Streamlit + SAM3).

Produces, for RASTER regions embedded in scientific diagrams (photos, icons,
logos, illustrations, screenshots, charts -- NOT structural diagram nodes
like boxes/arrows), a human-confirmed point plus a human-confirmed pixel
mask per instance. This supervision feeds two fine-tuning efforts: Molmo
(raster-region pointing) and SAM3 (raster-region masking).

SAM3 is used exclusively through its point/box *interactive* interface
(`Sam3TrackerModel`, the SAM2-compatible visual-prompting mode) -- never the
text/concept-prompted `Sam3Model`. A raster region has no consistent
open-vocabulary description across diagrams, so text prompting does not
work for it; a human confirms every point and every mask instead.

Modules:
- datamodel: dataclasses, the instance state machine, RLE mask helpers.
- store:     one-JSON-per-image persistence with file locking.
- sam3_backend: embed-once-per-image / segment-per-click SAM3 wrapper.
- compositor: PIL rendering of base image + mask + point layers.
- app:       the Streamlit review app (one canvas, points + masks together).
- ingest:    offline Molmo proposal script (NOT part of the app).
"""
