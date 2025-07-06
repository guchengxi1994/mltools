from mltools.log.logger import logger

logger.info(
    """
    __  _____  ______            __    
   /  |/  / / /_  __/___  ____  / /____
  / /|_/ / /   / / / __ \/ __ \/ / ___/
 / /  / / /___/ / / /_/ / /_/ / (__  )
/_/  /_/_____/_/  \____/\____/_/____/
"""
)

__version__ = "0.1.2"

import multiprocessing

__CPUS__ = multiprocessing.cpu_count()
del multiprocessing

import os


def split_file_name(p: str) -> str:
    s = os.path.split(p)[-1]
    return os.path.splitext(s)[0]


def exists(s: str, ls: list) -> bool:
    res = [s in ls for v in ls]
    return len(res) > 0
