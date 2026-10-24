"""Bridge BISINDO — perantara antara ESP32-CAM, service Docker, dan web app.

Paket ini berisi:
- ``config``    : pembacaan konfigurasi dari environment / ``.env``
- ``upstream``  : klien HTTP ke service handlandmarker & model
- ``pipeline``  : buffer frame, pengodean video, dan orkestrasi prediksi
- ``app``       : aplikasi FastAPI (entry point)
"""

__all__ = ["config", "upstream", "pipeline", "app"]
