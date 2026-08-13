import torch
from torch import nn
import torch.distributed as dist

from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


class NaiveDDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):	
        super().__init__()
        self.module = module

        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(
        self,
        sharded: torch.Tensor,
    ):
        out = self.module(sharded)
        return out
    
    def all_reduce_gradients(self):
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, async_op=False)

    def average_gradients(self, state=None):
        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad.div_(self.world_size)
    
    def finish_gradient_synchronization(self):
        self.all_reduce_gradients()
        self.average_gradients()


class FlatDDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
    ):	
        super().__init__()
        self.module = module

        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def forward(
        self,
        sharded: torch.Tensor,
    ):
        out = self.module(sharded)
        return out
    
    def all_reduce_gradients(self):
        grads_view = [
            p.grad for p in self.module.parameters() if p.grad is not None
        ]

        flatten = _flatten_dense_tensors(grads_view)

        dist.all_reduce(flatten)
        return flatten, grads_view
    
    @torch.no_grad()
    def average_gradients(self, state):
        flat_grad, grads = state

        # 只发起一次大型除法 kernel
        flat_grad.div_(self.world_size)

        reduced_views = _unflatten_dense_tensors(flat_grad, grads)

        for grad, reduced_grad in zip(grads, reduced_views):
            grad.copy_(reduced_grad)

    
    def finish_gradient_synchronization(self):
        state = self.all_reduce_gradients()
        self.average_gradients(state)

class OverlapDDP(nn.Module):
    def __init__(
        self, 
        module: torch.nn.Module,
    ):
        super().__init__()
        self.module = module

        self.local_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.hook_)
                

    def forward(
        self,
        inputs,
    ):
        return self.module(inputs)

    def hook_(
        self,
        param,
    ):
        
        handle = dist.all_reduce(param.grad, async_op=True)
        self.handles.append([handle, param.grad])	
    
    @torch.no_grad()
    def finish_gradient_synchronization(self):
        
        for handle, grad in self.handles:
            handle.wait()
            grad.div_(self.world_size)
        # 异步通信记得句柄用完要清空，不然一直持有旧的grad无法释放最后会导致OOM
        self.handles.clear()




            

            
        
    


