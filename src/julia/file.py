from pathlib import Path

S_DIR = Path("src/julia/S")
WE_DIR = Path("src/julia/WE")


def S_file(m: int):
    return S_DIR / f"{m}.bin"


def WE_dir(m: int):
    return WE_DIR / str(m)
