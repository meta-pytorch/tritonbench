import kraken

from torch.distributed._functional_collectives import all_gather_tensor
from tritonbench.utils.distributed_op import DistributedOperator, register_benchmark


def torch_symm_mem_ag_mm(a_shared, b):
    a_gathered, c = torch.ops.symm_mem.fused_all_gather_matmul(
        a_shared, [b], gather_dim=0, group_name=dist.group.WORLD.group_name
    )
    return a_gathered, c[0]


class Operator(DistributedOperator):

    @register_benchmark()
    def torch(self, a_shared, b) -> None:
        return lambda: torch_symm_mem_ag_mm(a_shared, b)

    @register_benchmark()
    def triton(self, a_shared, b) -> None:
        return lambda: kraken.fused.all_gather_matmul(a_shared, b)

    @register_benchmark()
    def nccl(self, a_shared, b) -> None:
        def _inner():
            a_gathered = all_gather_tensor(a_shared, 0, "0")
            return a_gathered, torch.matmul(a_gathered, b)

        return lambda: _inner()
