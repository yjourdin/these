from multiprocessing import set_start_method
from runpy import run_module

set_start_method("forkserver")

run_module("src.main")
