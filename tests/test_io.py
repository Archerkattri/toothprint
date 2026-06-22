"""Medical-format I/O: round-trip every format a dentist uses + the security guards."""

import numpy as np
import pytest

from toothprint import io
from toothprint.io import _limits
from toothprint.io.detect import detect, RADIOGRAPH, SCAN, VOLUME

trimesh = pytest.importorskip("trimesh")
pydicom = pytest.importorskip("pydicom")
nibabel = pytest.importorskip("nibabel")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


# --- fixtures: synthesize tiny real files ----------------------------------


def _box(tmp_path, ext):
    p = tmp_path / f"m.{ext}"
    trimesh.creation.box(extents=[10, 12, 8]).export(str(p))
    return p


def _img(tmp_path, ext, shape=(40, 48)):
    arr = (np.random.default_rng(0).random(shape) * 255).astype(np.uint8)
    p = tmp_path / f"x.{ext}"
    Image.fromarray(arr).save(str(p))
    return p, arr


def _dicom(tmp_path, photometric="MONOCHROME2", spacing=(0.1, 0.1)):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        generate_uid,
        SecondaryCaptureImageStorage,
    )

    arr = (np.random.default_rng(1).integers(0, 4096, (40, 48))).astype(np.uint16)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    p = tmp_path / "x.dcm"
    ds = FileDataset(str(p), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated, ds.BitsStored, ds.SamplesPerPixel = 16, 12, 1
    ds.PhotometricInterpretation = photometric
    ds.PixelRepresentation, ds.Modality = 0, "IO"
    if spacing:
        ds.ImagerPixelSpacing = list(spacing)
    ds.PixelData = arr.tobytes()
    ds.save_as(str(p), enforce_file_format=True)
    return p, arr


def _nifti(tmp_path, name="v.nii.gz", shape=(12, 14, 10)):
    vol = np.random.default_rng(2).random(shape).astype(np.float32)
    p = tmp_path / name
    nibabel.save(nibabel.Nifti1Image(vol, np.diag([0.3, 0.3, 0.3, 1])), str(p))
    return p, vol


# --- meshes ----------------------------------------------------------------


@pytest.mark.parametrize("ext", ["stl", "ply", "obj", "off", "glb"])
def test_load_scan_formats(tmp_path, ext):
    s = io.load(str(_box(tmp_path, ext)))
    assert isinstance(s, io.Scan) and s.n_vertices > 0 and s.n_faces == 12
    assert s.vertices.shape[1] == 3 and np.isfinite(s.vertices).all()


def test_load_3mf(tmp_path):
    pytest.importorskip("lxml")
    s = io.load(str(_box(tmp_path, "3mf")))
    assert isinstance(s, io.Scan) and s.n_faces == 12


def test_scan_to_open3d_mesh_and_cloud(tmp_path):
    pytest.importorskip("open3d")
    s = io.load(str(_box(tmp_path, "stl")))
    assert s.to_open3d() is not None
    cloud = io.Scan(vertices=s.vertices, faces=None, source_format="x")
    assert cloud.n_faces == 0 and cloud.to_open3d() is not None


# --- radiographs -----------------------------------------------------------


@pytest.mark.parametrize("ext", ["png", "jpg", "tif", "bmp"])
def test_load_radiograph_raster(tmp_path, ext):
    p, arr = _img(tmp_path, ext)
    r = io.load(str(p))
    assert isinstance(r, io.Radiograph) and r.shape == arr.shape
    assert 0.0 <= r.normalized.min() and r.normalized.max() <= 1.0


def test_detect_raster_by_extension_fallback(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"no recognizable magic here at all")
    assert detect(str(p)) == ("png", RADIOGRAPH)  # unknown magic, known extension


def test_radiograph_color_to_luminance(tmp_path):
    p = tmp_path / "c.png"
    Image.fromarray(
        (np.random.default_rng(0).random((30, 30, 3)) * 255).astype(np.uint8)
    ).save(str(p))
    assert io.load(str(p)).pixels.ndim == 2


def test_load_dicom_monochrome1_inverts_and_spacing(tmp_path):
    p, arr = _dicom(tmp_path, photometric="MONOCHROME1", spacing=(0.1, 0.12))
    r = io.load(str(p))
    assert r.source_format == "dicom" and r.pixel_spacing_mm == 0.1
    assert r.photometric == "MONOCHROME1" and r.modality == "IO"


def test_load_dicom_monochrome2(tmp_path):
    p, _ = _dicom(tmp_path, photometric="MONOCHROME2", spacing=None)
    r = io.load(str(p))
    assert r.pixel_spacing_mm is None and r.shape == (40, 48)


def test_normalized_constant_image_is_zero():
    r = io.Radiograph(
        pixels=np.ones((4, 4), np.float32), pixel_spacing_mm=None, source_format="x"
    )
    assert np.allclose(r.normalized, 0.0)


# --- volumes ---------------------------------------------------------------


def test_load_nifti(tmp_path):
    p, vol = _nifti(tmp_path)
    v = io.load(str(p))
    assert isinstance(v, io.Volume) and v.shape == vol.shape
    assert all(abs(s - 0.3) < 1e-4 for s in v.spacing_mm)


def test_load_nifti_plain_nii(tmp_path):
    p, vol = _nifti(tmp_path, name="v.nii")
    assert io.load_volume(str(p)).shape == vol.shape


def test_load_dicom_series(tmp_path):
    d = tmp_path / "series"
    d.mkdir()
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        generate_uid,
        SecondaryCaptureImageStorage,
    )

    for k in range(4):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(d / f"s{k}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Rows, ds.Columns = 16, 16
        ds.BitsAllocated, ds.BitsStored, ds.SamplesPerPixel = 16, 12, 1
        ds.PhotometricInterpretation, ds.PixelRepresentation = "MONOCHROME2", 0
        ds.ImagePositionPatient = [0.0, 0.0, float(k)]
        ds.PixelSpacing = [0.2, 0.2]
        ds.SliceThickness = 0.5
        ds.PixelData = (np.ones((16, 16), np.uint16) * k).tobytes()
        ds.save_as(str(d / f"s{k}.dcm"), enforce_file_format=True)
    (d / "notdicom.txt").write_text("ignore me")
    (d / "subdir").mkdir()  # non-file entry is skipped
    v = io.load(str(d))
    assert v.shape == (4, 16, 16) and v.spacing_mm == (0.5, 0.2, 0.2)
    assert v.meta["n_slices"] == 4


# --- detection + dispatch --------------------------------------------------


def test_magic_beats_extension(tmp_path):
    p, _ = _img(tmp_path, "png")
    fake = tmp_path / "fake.stl"
    fake.write_bytes(p.read_bytes())
    assert isinstance(io.load(str(fake)), io.Radiograph)  # PNG magic wins


def test_detect_categories(tmp_path):
    assert detect(str(_box(tmp_path, "stl")))[1] == SCAN
    assert detect(str(_img(tmp_path, "png")[0]))[1] == RADIOGRAPH
    assert detect(str(_nifti(tmp_path)[0]))[1] == VOLUME


def test_ascii_stl_detected(tmp_path):
    p = tmp_path / "a.stl"
    trimesh.creation.box().export(str(p), file_type="stl_ascii")
    assert detect(str(p)) == ("stl", SCAN)


# --- security guards -------------------------------------------------------


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "e.png"
    p.write_bytes(b"")
    with pytest.raises(io.CorruptFile):
        io.load(str(p))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(io.CorruptFile):
        io.load(str(tmp_path / "nope.stl"))


def test_unsupported_format_rejected(tmp_path):
    p = tmp_path / "j.xyz"
    p.write_bytes(b"\x00\x01garbage")
    with pytest.raises(io.UnsupportedFormat):
        io.load(str(p))


def test_oversize_file_rejected(tmp_path, monkeypatch):
    p, _ = _img(tmp_path, "png")
    monkeypatch.setattr(_limits, "MAX_FILE_BYTES", 8)
    with pytest.raises(io.FileTooLarge):
        io.load(str(p))


def test_vertex_cap_enforced(tmp_path, monkeypatch):
    p = _box(tmp_path, "stl")
    monkeypatch.setattr(_limits, "MAX_MESH_VERTICES", 3)
    with pytest.raises(io.FileTooLarge):
        io.load_scan(str(p))


def test_face_cap_enforced(tmp_path, monkeypatch):
    p = _box(tmp_path, "stl")
    monkeypatch.setattr(_limits, "MAX_MESH_FACES", 3)
    with pytest.raises(io.FileTooLarge):
        io.load_scan(str(p))


def test_pixel_cap_enforced(tmp_path, monkeypatch):
    p, _ = _img(tmp_path, "png")
    monkeypatch.setattr(_limits, "MAX_IMAGE_PIXELS", 4)
    with pytest.raises(io.FileTooLarge):
        io.load(str(p))


def test_voxel_cap_enforced(tmp_path, monkeypatch):
    p, _ = _nifti(tmp_path)
    monkeypatch.setattr(_limits, "MAX_VOLUME_VOXELS", 4)
    with pytest.raises(io.FileTooLarge):
        io.load(str(p))


def test_corrupt_mesh_rejected(tmp_path):
    p = tmp_path / "bad.ply"
    p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 999\nend_header\n")
    with pytest.raises(io.IOError_):
        io.load_scan(str(p))


def test_corrupt_dicom_rejected(tmp_path):
    p = tmp_path / "bad.dcm"
    p.write_bytes(b"\x00" * 128 + b"DICM" + b"\xff" * 64)
    with pytest.raises(io.IOError_):
        io.load(str(p))


def test_wrong_category_raises(tmp_path):
    with pytest.raises(io.UnsupportedFormat):
        io.load_radiograph(str(_box(tmp_path, "stl")))
    with pytest.raises(io.UnsupportedFormat):
        io.load_scan(str(_img(tmp_path, "png")[0]))


def test_load_volume_rejects_non_nifti(tmp_path):
    with pytest.raises(io.UnsupportedFormat):
        io.load_volume(str(_box(tmp_path, "stl")))


def test_dicom_series_requires_directory(tmp_path):
    with pytest.raises(io.UnsupportedFormat):
        io.load_dicom_series(str(_box(tmp_path, "stl")))


def test_dicom_series_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "x.txt").write_text("nothing")
    with pytest.raises(io.CorruptFile):
        io.load_dicom_series(str(d))


def test_non_finite_vertices_rejected(tmp_path, monkeypatch):
    p = _box(tmp_path, "ply")
    import trimesh as tm

    real = tm.load

    def fake(*a, **k):
        m = real(*a, **k)
        m.vertices[0] = [np.inf, 0, 0]
        return m

    monkeypatch.setattr(tm, "load", fake)
    with pytest.raises(io.CorruptFile):
        io.load_scan(str(p))


# --- adversarial / fuzz: defensive parser branches -------------------------


def test_dicom_unreadable_bytes(tmp_path):
    p = tmp_path / "x.dcm"
    p.write_bytes(b"\x12\x34not a dicom at all" * 4)
    with pytest.raises(io.CorruptFile):
        io.load(str(p))


def test_dicom_color_to_luminance(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        generate_uid,
        SecondaryCaptureImageStorage,
    )

    arr = (np.random.default_rng(3).integers(0, 256, (20, 24, 3))).astype(np.uint8)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    p = tmp_path / "rgb.dcm"
    ds = FileDataset(str(p), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = 20, 24
    ds.BitsAllocated, ds.BitsStored, ds.SamplesPerPixel = 8, 8, 3
    ds.PhotometricInterpretation, ds.PixelRepresentation = "RGB", 0
    ds.PlanarConfiguration = 0
    ds.PixelData = arr.tobytes()
    ds.save_as(str(p), enforce_file_format=True)
    assert io.load(str(p)).pixels.ndim == 2


def test_dicom_no_extension_detected_as_volume_then_loaded(tmp_path):
    p, _ = _dicom(tmp_path)
    noext = tmp_path / "anon0001"
    noext.write_bytes(p.read_bytes())
    assert detect(str(noext)) == ("dicom", VOLUME)
    assert isinstance(io.load(str(noext)), io.Radiograph)  # single file -> radiograph


def test_corrupt_tiff_rejected(tmp_path):
    p = tmp_path / "b.tif"
    p.write_bytes(b"II*\x00" + b"\xff" * 40)
    with pytest.raises(io.IOError_):
        io.load(str(p))


def test_corrupt_png_rejected(tmp_path):
    p = tmp_path / "b.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    with pytest.raises(io.IOError_):
        io.load(str(p))


def test_pil_decompression_bomb_rejected(tmp_path, monkeypatch):
    p, _ = _img(tmp_path, "png", shape=(80, 80))
    monkeypatch.setattr(_limits, "MAX_IMAGE_PIXELS", 16)  # PIL itself raises the bomb
    with pytest.raises(io.FileTooLarge):
        io.load(str(p))


@pytest.mark.parametrize(
    "magic,ext,expect",
    [
        (b"GIF89a" + b"\x00" * 20, "gif", "gif"),
        (b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8, "webp", "webp"),
        (b"OFF\n0 0 0\n", "off", "off"),
    ],
)
def test_detect_minor_formats(tmp_path, magic, ext, expect):
    p = tmp_path / f"f.{ext}"
    p.write_bytes(magic)
    fmt, _ = detect(str(p))
    assert fmt == expect


def test_nifti_detected_by_magic_without_extension(tmp_path):
    p, _ = _nifti(tmp_path, name="vol.nii")
    raw = p.read_bytes()
    q = tmp_path / "scan_blob"
    q.write_bytes(raw)
    assert detect(str(q)) == ("nifti", VOLUME)


def test_3mf_not_a_zip_rejected(tmp_path):
    p = tmp_path / "bad.3mf"
    p.write_bytes(b"PK\x03\x04 but truncated zip")
    with pytest.raises(io.IOError_):
        io.load_scan(str(p))


def test_3mf_zip_bomb_rejected(tmp_path, monkeypatch):
    pytest.importorskip("lxml")
    p = _box(tmp_path, "3mf")
    monkeypatch.setattr(_limits, "MAX_DECOMPRESSED_BYTES", 4)
    with pytest.raises(io.FileTooLarge):
        io.load_scan(str(p))


def test_ply_point_cloud_no_faces(tmp_path):
    p = tmp_path / "cloud.ply"
    pts = np.random.default_rng(0).random((50, 3)).astype(np.float32)
    trimesh.PointCloud(pts).export(str(p))
    s = io.load_scan(str(p))
    assert s.faces is None and s.n_faces == 0 and s.n_vertices == 50


def test_nifti_gz_bomb_guard(tmp_path, monkeypatch):
    p, _ = _nifti(tmp_path)
    monkeypatch.setattr(_limits, "MAX_DECOMPRESSED_BYTES", 4)
    with pytest.raises(io.FileTooLarge):
        io.load(str(p))


def test_corrupt_nifti_rejected(tmp_path):
    p = tmp_path / "bad.nii"
    p.write_bytes(b"\x00" * 10 + b"garbage nifti header")
    with pytest.raises(io.IOError_):
        io.load_volume(str(p))


def test_dicom_series_size_cap(tmp_path, monkeypatch):
    d = tmp_path / "series"
    d.mkdir()
    p, _ = _dicom(tmp_path)
    (d / "s0.dcm").write_bytes(p.read_bytes())
    monkeypatch.setattr(_limits, "MAX_DECOMPRESSED_BYTES", 4)
    with pytest.raises(io.FileTooLarge):
        io.load_dicom_series(str(d))


def _bare_dicom(tmp_path, name):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        generate_uid,
        SecondaryCaptureImageStorage,
    )

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    p = tmp_path / name
    ds = FileDataset(str(p), {}, file_meta=meta, preamble=b"\0" * 128)
    return p, ds


def test_dicom_without_pixel_data_rejected(tmp_path):
    p, ds = _bare_dicom(tmp_path, "nopix.dcm")
    ds.Rows, ds.Columns, ds.Modality = 16, 16, "IO"
    ds.save_as(str(p), enforce_file_format=True)
    with pytest.raises(io.CorruptFile):
        io.load(str(p))


def test_dicom_multiframe_grayscale_takes_first(tmp_path):
    p, ds = _bare_dicom(tmp_path, "mf.dcm")
    frames = np.random.default_rng(5).integers(0, 4096, (3, 20, 24)).astype(np.uint16)
    ds.Rows, ds.Columns = 20, 24
    ds.NumberOfFrames = 3
    ds.BitsAllocated, ds.BitsStored, ds.SamplesPerPixel = 16, 12, 1
    ds.PhotometricInterpretation, ds.PixelRepresentation = "MONOCHROME2", 0
    ds.PixelData = frames.tobytes()
    ds.save_as(str(p), enforce_file_format=True)
    assert io.load(str(p)).shape == (20, 24)


def test_detect_directory_rejected(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(io.UnsupportedFormat):
        detect(str(d))  # a directory is not a regular file
