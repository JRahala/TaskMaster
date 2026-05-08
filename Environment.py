from typing import Iterator, List, Sequence
from dataclasses import dataclass, field
from Core import TaskInstance, TaskGenerator

@dataclass(frozen=True)
class Timestep:
    t: int
    task_instances: List[TaskInstance]

    def __repr__(self):
        return f"Timestep(t={self.t}, task_instances={len(self.task_instances)} tasks)"

class Environment:
    def __init__(
        self,
        generators,
        timesteps
    ):
        self.generators = list(generators)
        self.timesteps = timesteps
        self._history = self._build_history()
        self._all_tasks = [task for timestep in self._history for task in timestep.task_instances]

    def _build_history(self):
        for generator in self.generators:
            generator.reset()
        
        history = []
        for timestep in range(self.timesteps):
            task_instances = []
            for generator in self.generators:
                task_instance = generator.create_instance(creation_time=timestep)
                if task_instance: task_instances.append(task_instance)
            history.append(Timestep(t=timestep, task_instances=task_instances))
        return history
    
    def get_timestep(self, timestep):
        if 0 > timestep or timestep >= self.timesteps: raise Exception("Timestep out of range")
        return self._history[timestep]
    
    def all_tasks(self):
        return self._all_tasks
    
    def __iter__(self):
        return iter(self._history)
    
    def __len__(self):
        return self.timesteps
    
    def __repr__(self):
        return "Environment(timesteps={self.timesteps},generators={self.generators})"