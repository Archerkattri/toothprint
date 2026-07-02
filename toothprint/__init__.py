"""ToothPrint — certified dental-imaging intelligence.

Three capabilities over one dental signal:

  * ``toothprint.identity``  — recognise a person by their teeth (3D scans + 2D radiographs)
  * ``toothprint.change``    — certify whether a radiograph bone-level change is real
  * ``toothprint.surface``   — certify whether a 3D surface change is real

The certification logic, registration, and identification cores depend only on
numpy / scipy / opencv / open3d. Learned front-ends (tooth detection,
photogrammetric reconstruction) are pluggable and optional.
"""

__version__ = "1.1.0"
