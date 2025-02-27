from pathlib import Path

PARENT_DIR = Path(__file__).parent

S_DIR = PARENT_DIR / "S"
WE_DIR = PARENT_DIR / "WE"


def S_file(m: int):
    return S_DIR / f"{m}.bin"


def WE_dir(m: int):
    return WE_DIR / str(m)
