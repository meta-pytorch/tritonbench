# Scripts that load operators and generate the metadata
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


from ..common import setup_tritonbench_cwd

setup_tritonbench_cwd()

from mpp.applications import MPPAnalysis
from tritonbench.operators import load_opbench_by_name
from tritonbench.utils.parser import get_parser


def run_one_operator():
    args = [
        "--op",
        "gdpa",
        "--only",
        "gdpa",
        "--sparsity",
        "1.0",
        "--head",
        "4",
        "--dim",
        "512",
        "--max_seq_len",
        "1000",
        "--batch",
        "1152",
        "--rep",
        "3000",
        "--sleep",
        "1.0",
    ]
    parser = get_parser(args)
    tb_args, extra_args = parser.parse_known_args(args)
    opbench_cls = load_opbench_by_name(tb_args.op)
    opbench = opbench_cls(
        tb_args=tb_args,
        extra_args=extra_args,
    )
    opbench.run()


if __name__ == "__main__":
    MPPAnalysis(run_one_operator).cli()
