import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from dataclasses import dataclass
from cs336_basics.model import Embedding, Linear
from typing import Dict
from torch import Tensor

@dataclass
class FSDPParamState:
    # 记录所有模块shard需要的参数
    master_shard: torch.Tensor
    original_shape: torch.Size
    original_numel: int
    shard_numel: int
    padding_size: int
    

    # 下面是运行期间的临时状态
    communication_shard: torch.Tensor | None = None
    gathered_buffer: torch.Tensor | None = None
    full_weight: torch.Tensor | None = None
    gather_work: object | None = None



class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_type: torch.dtype | None = None,
    ):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_type = compute_type
        self.module = module

        self.modules_states: Dict[nn.Module, FSDPParamState] = {}
        self.params_states: Dict[Tensor, FSDPParamState] =  {}
        

        for submodule in module.modules():
            # 因为这里可能在根module上，根module也不属于Linear和Embedding，
            # 所以不能用if-else判断
            if not isinstance(submodule, (Linear, Embedding)):
                continue

            state = self._shard_parameters(submodule.weight)
            self.modules_states[submodule] = state
            self.params_states[submodule.weight] = state
            submodule.weight.data = state.master_shard
            submodule.weight.grad_dtype = torch.float32

            submodule.register_forward_pre_hook(
                self._forward_pre_hook
            )

            submodule.register_forward_hook(
                self._forward_pos_hook
            )
            submodule.weight.register_post_accumulate_grad_hook(
                self._sharded_grad_hook
            )  

            if isinstance(submodule, Linear):  
                submodule.register_full_backward_pre_hook(
                    self._linear_backward_pre_hook
                )

        for param in module.parameters():
            if param.requires_grad and param not in self.params_states:
                param.register_post_accumulate_grad_hook(
                    self._replicated_grad_hook
                )

    

    def forward(
        self,
        *inputs,
        **kwargs,
    ):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        pass

    def _shard_parameters(
        self,
        weight,
    ):  
        original_shape = weight.shape
        original_numel = weight.numel()
        shard_numel = (original_numel + (self.world_size - 1)) // self.world_size
        padding_size = shard_numel * self.world_size - original_numel

        flat_weight = weight.detach().float().reshape(-1)
        padded = F.pad(flat_weight, (0, padding_size))
        start = self.rank * shard_numel
        end = (self.rank + 1) * shard_numel
        shard_weight = padded[start:end].clone()

        return FSDPParamState(
            master_shard=shard_weight,
            original_shape=original_shape, 
            original_numel=original_numel,
            shard_numel=shard_numel,
            padding_size=padding_size,
        )

    def _all_gather_weight(
        self,
        module: nn.Module,
    ):
        state = self.modules_states[module]
        compute_type = self.compute_type


        communication_shard = state.master_shard

        if compute_type is not None:
            communication_shard = communication_shard.to(compute_type)

        gathered = torch.empty(
            state.shard_numel * self.world_size,
            dtype=communication_shard.dtype,
            device=communication_shard.device,
        )

        dist.all_gather_into_tensor(
            gathered,
            communication_shard,
            async_op=False,
        )

        full_weight = gathered[:state.original_numel]
        full_weight = full_weight.reshape(state.original_shape)

        module.weight.data = full_weight

    def _forward_pre_hook(
        self,
        module,
        args,
    ):
         self._all_gather_weight(module)
    
    def _forward_pos_hook(
        self,
        module,
        args,
        output,
    ):
        # Embedding 的输入 token IDs 不参与求导，因此在初始化的时候直接绑定register_full_backward_pre_hook会
        # 报warning，因此在把这这个hook绑定到grad_output上。
        if (    
            isinstance(module, Embedding)
            and isinstance(output, torch.Tensor)
            and output.requires_grad
        ):
            output.register_hook(
                lambda grad, module=module:
                    self._embedding_output_grad_hook(module, grad)
            )

        state = self.modules_states[module]

        # 前向完成，释放完整 weight。
        module.weight.data = state.master_shard
        
       
    def _linear_backward_pre_hook(
        self,
        module,
        grad_output,
    ):
        self._all_gather_weight(module)


    def _embedding_output_grad_hook(
        self,
        module,
        grad,
    ):
        # output grad 到达后、Embedding backward 执行前恢复完整 weight。
        self._all_gather_weight(module)

    
    def _sharded_grad_hook(
        self,
        param,
    ):
        state = self.params_states[param]

        flat_grad = param.grad.detach().reshape(-1)
        compute_type = self.compute_type

        communication_grad = F.pad(
            flat_grad,
            (0, state.padding_size),
        )

        if compute_type is not None:
            communication_grad = communication_grad.to(compute_type)

        local_grad = torch.empty(
            state.shard_numel,
            dtype=communication_grad.dtype,
            device=communication_grad.device,
        )

        dist.reduce_scatter_tensor(
            local_grad,
            communication_grad,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )

        # 每个 rank 的 loss 是本地 mean，因此对 rank 梯度求平均。
        local_grad = local_grad.to(torch.float32)
        local_grad.div_(self.world_size)

        # 恢复 FP32 master shard 及其对应的本地梯度。
        param.data = state.master_shard
        param.grad.data = local_grad

    def _replicated_grad_hook(
        self,
        param,
    ):
        # Norm 等参数没有分片，但不同 rank 使用不同数据，
        # 因此仍需要进行 DDP 式梯度平均。
        dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=False,
        )

        param.grad.div_(self.world_size)

    # 测试接口
    def _gather_full_params_for_test(self):
        result = {}

        for name, param in self.module.named_parameters():
            if param in self.params_states:
                state = self.params_states[param]
                communication_shard = state.master_shard
                
                gathered = torch.empty(
                    state.shard_numel * self.world_size,
                    dtype=communication_shard.dtype,
                    device=communication_shard.device,
                )
                
                dist.all_gather_into_tensor(
                    gathered,
                    communication_shard,
                    async_op=False,
                )

                original = gathered[:state.original_numel]
                original = original.reshape(state.original_shape)
                
                result[name] = original
            else:
                result[name] = param.detach().clone()

        return result
        




        


        

    
