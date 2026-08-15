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
            if isinstance(submodule, (Linear, Embedding)):
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
                    self._backward_pos_hook
                )    
                submodule.register_full_backward_pre_hook(
                    self._backward_pre_hook
                )

        for param in module.parameters():
            if param.requires_grad and param not in self.params_states:
                param.register_post_accumulate_grad_hook(
                    self._normal_grad_hook
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


    def _forward_pre_hook(
        self,
        module,
        args,
    ):
        state = self.modules_states[module]
        
        communication_shard = state.master_shard
        if self.compute_type is not None:
            communication_shard = communication_shard.to(self.compute_type)

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

        module.weight.data = original
        
    
    def _forward_pos_hook(
        self,
        module,
        args,
        output,
    ):
        state = self.modules_states[module]
        module.weight.data = state.master_shard
        
       
    def _backward_pre_hook(
        self,
        module,
        grad_output,
    ):
        state = self.modules_states[module]
        communication_shard = state.master_shard
        if self.compute_type is not None:
            communication_shard = communication_shard.to(self.compute_type)


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

        module.weight.data = original
    
    def _backward_pos_hook(
        self,
        param,
    ):
        state = self.params_states[param]
        
        flat_grad = param.grad.detach().reshape(-1) 
        communication_grad = F.pad(
            flat_grad,
            (0, state.padding_size),
        )


        if self.compute_type is not None:
            communication_grad = communication_grad.to(self.compute_type)
        
        local_grad = torch.empty_like(
            state.master_shard,
            dtype=communication_grad.dtype,
            device=communication_grad.device,
        )

        dist.reduce_scatter_tensor(
            local_grad,
            communication_grad,
            op= dist.ReduceOp.SUM,
            async_op=False,
        )

        local_grad = local_grad.float()
        local_grad.div_(self.world_size)

        param.grad.data = local_grad
        param.data = state.master_shard

    
    def _normal_grad_hook(
        self,
        param,
    ):
        dist.all_reduce(
            param.grad,
            op= dist.ReduceOp.SUM,
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
        




        


        

    
