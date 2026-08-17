import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from dataclasses import dataclass
from cs336_basics.model import Embedding, Linear
from typing import Dict, Any
from torch import Tensor

@dataclass
class FSDPParamState:
    # 记录所有模块shard需要的参数
    master_shard: torch.Tensor
    original_shape: torch.Size
    original_numel: int
    shard_numel: int
    padding_size: int
    
    # 下面是运行期间的临时状态，异步的时候需要这些状态
    # 来判断当前需要执行的逻辑。
    communication_shard: torch.Tensor | None = None
    gathered_buffer: torch.Tensor | None = None
    #full_weight: torch.Tensor | None = None
    gather_work: Any = None

    
    communication_gard: torch.Tensor | None = None
    local_grad: Any = None
    grad_work: Any = None


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

        # 持有reduce scatter的async handle
        self.pending_grad_sync = []

        self.modules_states: Dict[nn.Module, FSDPParamState] = {}
        self.params_states: Dict[Tensor, FSDPParamState] =  {}

        # 保存需要shard paramters的模块的引用
        sharded_modules = [
            submodule
            for submodule in module.modules()
            if isinstance(submodule, (Linear, Embedding))
        ]
        
        for submodule in sharded_modules:
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

        # 注意，这里构建prefetch的时序依赖模块注册时候的顺序，必须保证forward
        # 的执行顺序和注册顺序一致，这里的构建才是正确的。对应model.py下的模块
        # 已经修改保证了这个逻辑。在这里提前构建好预取的映射后，hook就可以简单
        # 的找到需要提前预取的目标了。
        self.forward_prefetch_targets = {
            sharded_modules[i]: sharded_modules[i + 2]
            for i in range(len(sharded_modules) - 2)
        }

    

    def forward(
        self,
        *inputs,
        **kwargs,
    ):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        world_size = self.world_size
        for handle, param in self.pending_grad_sync:
            handle.wait()

            state = self.params_states.get(param)

            if state is not None:
                local_grad = state.local_grad
                local_grad = local_grad.to(torch.float32)
                local_grad.div_(world_size)

                # 恢复 FP32 master shard 及其对应的本地梯度。
                param.data = state.master_shard
                param.grad.data = local_grad

                # 释放对应的缓存
                state.local_grad = None
                state.communication_gard = None
                state.grad_work = None

            else:
                local_grad = param.grad
                local_grad.div_(world_size)
        
        self.pending_grad_sync.clear()

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
        async_op: bool,
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

        work = dist.all_gather_into_tensor(
            gathered,
            communication_shard,
            async_op=async_op,
        )

        # 同步版本直接把对应处理后的full weight给module weight data持有。
        # 异步版本要持有buffer和handle来保证通信过程的正确。
        if async_op:
            state.communication_shard = communication_shard
            state.gather_work = work
            state.gathered_buffer = gathered
            
        else:
            full_weight = gathered[:state.original_numel]
            full_weight = full_weight.reshape(state.original_shape)
            module.weight.data = full_weight
    
    def _process_and_prefetch(
        self,
        module,
    ):
        state = self.modules_states[module]
        
        # 句柄为空说明现在是流水线的填充状态
        if state.gather_work is None:
            self._all_gather_weight(
                module,
                async_op=True
            )

        handle = state.gather_work
        gathered = state.gathered_buffer

        if handle is None or gathered is None:
            raise RuntimeError("All-gather has not been started")
    
        handle.wait()

        full_weight = gathered[:state.original_numel]
        full_weight = full_weight.reshape(state.original_shape)
        module.weight.data = full_weight

        prefetch_target = self.forward_prefetch_targets.get(module)
        if prefetch_target is not None:
            self._all_gather_weight(prefetch_target, async_op=True)

        
    def _forward_pre_hook(
        self,
        module,
        args,
    ):
         self._process_and_prefetch(module)
    
    def _forward_pos_hook(
        self,
        module,
        args,
        output,
    ):
        # Embedding 的输入 token IDs 不参与求导，因此在初始化的时候直接绑定register_full_backward_pre_hook会
        # 报warning，因此在把embedding对应all gather绑定到前向的输出output上。
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

        # 前向完成，释放完整 weight, 同时应该释放state中持有的对应buffer
        module.weight.data = state.master_shard
        state.gather_work = None
        state.gathered_buffer = None
        state.communication_shard = None
        
    def _linear_backward_pre_hook(
        self,
        module,
        grad_output,
    ):
        self._all_gather_weight(module, async_op=False)


    def _embedding_output_grad_hook(
        self,
        module,
        grad,
    ):
        # output grad 到达后、Embedding backward 执行前恢复完整 weight。
        self._all_gather_weight(module, async_op=False)

    
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

        work = dist.reduce_scatter_tensor(
            local_grad,
            communication_grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        

        state.local_grad = local_grad
        state.communication_gard = communication_grad
        state.grad_work = work

        self.pending_grad_sync.append([work, param])
        # # 每个 rank 的 loss 是本地 mean，因此对 rank 梯度求平均。
        # local_grad = local_grad.to(torch.float32)
        # local_grad.div_(self.world_size)

        # # 恢复 FP32 master shard 及其对应的本地梯度。
        # param.data = state.master_shard
        # param.grad.data = local_grad

    def _replicated_grad_hook(
        self,
        param,
    ):
        # Norm 等参数没有分片，但不同 rank 使用不同数据，
        # 因此仍需要进行 DDP 式梯度平均。
        work = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        self.pending_grad_sync.append([work, param])

        #param.grad.div_(self.world_size)

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
        




        


        

    
