from torch.optim import Optimizer
from typing import Any, Type
import torch.multiprocessing as mp
import torch.distributed as dist

import torch


class SharedOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any
    ):
        defaults = dict(**kwargs)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        

        params_list = list(params)
        first_element = params_list[0]
        # 把params collections也转换成params group的形式
        if isinstance(first_element, dict) and "params" in first_element:
            params = params_list
        else:
            params = [
                {
                    "params": params_list
                }
            ]
        # 每个rank保存正确的完整参数引用用于后续同步的时候使用
        self.full_params = params
        local_params_groups = []
        # 保留一个param_to_rank映射用于后step的时候区分更新的职责
        param_to_rank  = {}
        self.make_local_shard_(params, local_params_groups, param_to_rank)
        self.local_params_groups = local_params_groups
        self.param_to_rank = param_to_rank

        
        self._initializing = True
        super().__init__(local_params_groups, defaults)
        self._initializing = False

        self.optimizer = optimizer_cls(local_params_groups, **kwargs)

    def make_local_shard_(
        self,
        params_groups,
        local_param_group,
        param_to_rank
    ):
        total_numel = sum(
            param.numel()
            for group in params_groups
            for param in group["params"]
        )

        avg_numel = (total_numel + self.world_size - 1) / self.world_size
        
        current_rank = 0
        current_numel = 0
        
        # 所有的rank维护一个统一的current_rank和current_numel累计逻辑，当current_rank
        # 指示到自己的时候，把这个分片保存到自己的local_param_group实现分片
        for group in params_groups:
            current_params = []
            for param in group["params"]:
                numel = param.numel()
                if(current_numel > avg_numel):
                    current_rank += 1
                    current_numel = 0

                if(current_rank == self.rank):
                    current_params.append(param)
                param_to_rank[param] = current_rank
                current_numel += numel
            
            if current_params:
                # 复制数组字典，避免修改到full_params
                local_group = dict(group)
                local_group["params"] = current_params
                local_param_group.append(local_group)  

    @torch.no_grad()
    def step(
        self,
        closure = None,
        **kwargs
    ):
        loss = None
        if closure is not None:
            loss = closure(**kwargs)

        self.optimizer.step()

        for group in self.full_params:
            for param in group["params"]:
                src = self.param_to_rank[param]
                dist.broadcast(param, src=src)
        return loss
        

    def add_param_group(
        self,
        param_group: dict[str, Any]
    ):
        if getattr(self, "_initializing", False):
            # 初始化的时候参数已经提前分片
            Optimizer.add_param_group(self, param_group)
            return
        
        local_param_group = {}

        self.make_local_shard_(
            param_group, 
            [local_param_group], 
            self.param_to_rank
        )
        
        self.optimizer.add_param_group(local_param_group)
        self.local_params_groups.append(local_param_group)
        


		
		

		
	