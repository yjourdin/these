from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe
from multiprocessing.connection import Connection, wait
from typing import cast

from src.methods import MethodEnum

from .....constants import SENTINEL
from ....dir import DIR
from ....task import FutureTask, result_list
from ....threads.task import task_thread
from ...elicitation.config import Config, MIPConfig
from ..directory import DirectoryGroupDecision
from ..fields import GroupParameters
from ..seeds import Seeds
from ..task import PreferencePathTask


def collective_thread(
    m: int,
    n_tr: int,
    Atr_id: int,
    ko: int,
    fixed_lex_order: bool,
    Mo_id: int,
    group_size: int,
    group: GroupParameters,
    Mi_id: int,
    n_bc: int,
    same_alt: bool,
    D_id: int,
    Mie: bool,
    Mie_config: MIPConfig | None,
    Mie_id: int,
    method: MethodEnum,
    config: Config,
    nb_Mcp: int,
    Mc_id: int,
    path: bool,
    P_id: int,
    time_left: int,
    time_left_per_it: int,
    it: int,
    seeds: Seeds,
    precede_futures: list[FutureTask],
):
    assert isinstance(DIR, DirectoryGroupDecision)
    result_list(precede_futures)

    DMS = range(group_size)

    sources: dict[Connection, set[int]] = {}
    futures = []

    with ThreadPoolExecutor() as thread_pool:
        for dm_id in DMS:
            main_connection, worker_connection = Pipe()
            sources[main_connection] = set()
            task = PreferencePathTask(
                m,
                n_tr,
                Atr_id,
                ko,
                fixed_lex_order,
                Mo_id,
                group_size,
                group,
                Mi_id,
                dm_id,
                n_bc,
                same_alt,
                D_id,
                Mie,
                Mie_config,
                Mie_id,
                method,
                config,
                nb_Mcp,
                Mc_id,
                path,
                P_id,
                it,
            )
            futures.append(
                thread_pool.submit(
                    task_thread,
                    task,
                    {
                        "seed": seeds.P[P_id],
                        "max_time": min(time_left, time_left_per_it),
                        "connection": worker_connection,
                    },
                    [],
                )
            )

        working_connections = list(sources.keys())

        while working_connections and not set.intersection(*sources.values()):
            for connection in cast(list[Connection], wait(working_connections)):
                if (source := connection.recv()) != SENTINEL:
                    sources[connection] |= source
                else:
                    working_connections.remove(connection)

        for connection in sources:
            connection.send(
                set.intersection(*sources.values()).pop()
                if working_connections
                else SENTINEL
            )
