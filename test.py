import argparse
import sys

import numpy as np
from pandas import read_csv
from scipy.stats import kendalltau

from performance_table.core import NormalPerformanceTable
from rmp.model import RMPModel
from srmp.model import SRMPModel

parser = argparse.ArgumentParser()
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("Mo", type=argparse.FileType("r"), help="Original model")
parser.add_argument("Me", type=argparse.FileType("r"), help="Elicited model")
parser.add_argument(
    "-r",
    "--result",
    default=sys.stdout,
    type=argparse.FileType("w"),
    help="Result file",
)

args = parser.parse_args()

A = NormalPerformanceTable(read_csv(args.A))

so = args.Mo.read()
if "capacities" in so:
    Mo = RMPModel.from_json(so)
elif "weights" in so:
    Mo = SRMPModel.from_json(so)
else:
    ValueError("Mo is not a valid model")

se = args.Me.read()
if "capacities" in se:
    Me = RMPModel.from_json(se)
elif "weights" in se:
    Me = SRMPModel.from_json(se)
else:
    ValueError("Me is not a valid model")


Ro = Mo.rank(A).data.to_numpy()
Re = Me.rank(A).data.to_numpy()

outranking_o = np.less.outer(Ro, Ro).astype("int64", copy=False)
outranking_e = np.less.outer(Re, Re).astype("int64", copy=False)

outranking_o = outranking_o - outranking_o.transpose()
outranking_e = outranking_e - outranking_e.transpose()

ind = np.triu_indices(len(Ro), 1)

test_fitness = np.equal(outranking_o[ind], outranking_e[ind]).sum() / len(ind[0])

kendall_tau = kendalltau(Ro, Re).statistic

args.result.write(
    f"{args.A.name},"
    f"{args.Mo.name},"
    f"{args.Me.name},"
    f"{test_fitness},"
    f"{kendall_tau}\n"
)
