#!/usr/bin/python3
"""Multiprocessing."""
import math
from collections.abc import Callable
from multiprocessing import cpu_count
from typing import Any

from mpire.pool import WorkerPool
from tqdm import tqdm

from dobble.utils.logger import logger


def _batch_func(process_func: Callable[..., None],
                list_kwargs: list[dict[str, Any]]) -> None:
    """Process a batch."""
    for kwargs in list_kwargs:
        process_func(**kwargs)


def multiprocess(process_func: Callable[..., None],
                 list_kwargs: list[dict[str, Any]],
                 tqdm_title: str | None = None,
                 n_jobs: int | None = None) -> None:
    """Parallelize the process of a given function on a list of inputs."""
    if tqdm_title is None:
        tqdm_title = process_func.__name__

    n_process = len(list_kwargs)

    if n_jobs is None:
        n_jobs = math.floor(0.8 * cpu_count())
    elif n_jobs < 0:
        n_jobs = cpu_count()

    n_jobs = min(n_jobs, cpu_count())

    logger.info(f"Use {n_jobs} cpus out of {cpu_count()}")

    if n_jobs == 1:
        for kwargs in tqdm(list_kwargs, desc=tqdm_title):
            process_func(**kwargs)
    else:
        # Chunk the list of arguments into N approximately equal batches
        batch_size = math.ceil(n_process / float(n_jobs))

        with WorkerPool(n_jobs=n_jobs) as pool:
            params = [(process_func, list_kwargs[i: i + batch_size])
                      for i in range(0, n_process, batch_size)]

            progress_bar_options = {"desc": tqdm_title, 'unit': "job"}

            pool.map_unordered(_batch_func, params,
                               progress_bar=True,
                               progress_bar_options=progress_bar_options)
