# Build on Windows: python -m PyInstaller --noconfirm --clean meeting_record.spec
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
    get_module_file_attribute,
)

project = Path(SPECPATH)

# sounddevice loads PortAudio through ctypes rather than a Python import.
# Preserve the wheel's relative directory, which sounddevice uses at runtime.
portaudio_dir = Path(get_module_file_attribute("sounddevice")).parent / "_sounddevice_data" / "portaudio-binaries"
portaudio = list(portaudio_dir.glob("libportaudio*.dll"))
if not portaudio:
    raise RuntimeError("The Windows sounddevice wheel with bundled PortAudio is required.")

binaries = [(str(path), "_sounddevice_data/portaudio-binaries") for path in portaudio]
for package in ("ctranslate2", "sherpa_onnx"):
    binaries += collect_dynamic_libs(package)

# Includes faster-whisper's VAD assets. User models and recordings are deliberately
# outside this bundle and must never be copied from the developer's computer.
datas = [(str(project / name), ".") for name in ("LICENSE", "README.md")]
datas += collect_data_files("faster_whisper")
for distribution in ("faster-whisper", "ctranslate2"):
    datas += copy_metadata(distribution)
# New sherpa-onnx wheels depend on sherpa-onnx-core, which installs DLLs into
# sherpa_onnx/lib (collected above). Preserve both distributions' metadata.
datas += copy_metadata("sherpa-onnx", recursive=True)

a = Analysis(
    [str(project / "app.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["ctranslate2", "sherpa_onnx"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MeetingRecord",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MeetingRecord",
)
