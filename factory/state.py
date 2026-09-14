import hashlib
import json
import shutil
from pathlib import Path
from .api import FactoryError
from .config import VERSION
from .core import file_hash, save_json


def signature(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def inside(directory, relative):
    path = directory / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or path.is_symlink()
        or not path.resolve().is_relative_to(directory.resolve())
    ):
        raise FactoryError("Çalışma kaydında geçersiz dosya yolu.")
    return path


class Checkpoints:
    def __init__(self, directory):
        self.directory = directory
        self.path = directory / "checkpoints.json"
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"schema": 2, "version": VERSION, "stages": {}}
        )
        if self.data.get("schema") != 2 or self.data.get("version") != VERSION:
            raise FactoryError("Devam kaydı başka sürüme ait.")

    def read(self, name, key):
        stage = self.data["stages"].get(name)
        if not stage or stage.get("signature") != key:
            return None
        if not stage.get("files"):
            raise FactoryError("Aşama dosyaları eksik.")
        for relative, expected in stage["files"].items():
            path = inside(self.directory, relative)
            if not path.is_file() or file_hash(path) != expected:
                raise FactoryError(f"Kayıt dosyası değişmiş/eksik: {relative}")
        print(f"Kayıtlı aşama kullanılıyor: {name}", flush=True)
        return stage["result"]

    def save(self, name, key, result, files):
        hashes = {
            str(p.relative_to(self.directory)): file_hash(
                inside(self.directory, str(p.relative_to(self.directory)))
            )
            for p in files
        }
        self.data["stages"][name] = {
            "signature": key,
            "result": result,
            "files": hashes,
        }
        save_json(self.path, self.data)


def resume(source, destination):
    source = source.resolve()
    manifest = json.loads((source / "run.json").read_text())
    if manifest.get("schema") != 2 or manifest.get("version") != VERSION:
        raise FactoryError("Yalnız bu sürümün çalışma kaydından devam edilebilir.")
    checkpoints = Checkpoints(source)
    for name, stage in checkpoints.data["stages"].items():
        checkpoints.read(name, stage["signature"])
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise FactoryError("Devam kaydında sembolik bağlantı kabul edilmiyor.")
        if path.is_file() and (
            len(relative.parts) == 1
            or relative.parts[0]
            in {"events", "research", "audio", "voice_audition", "diagnostics"}
        ):
            target = inside(destination, str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return manifest
