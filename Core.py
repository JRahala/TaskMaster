import numpy as np

def no_penalty(overdue_time): return 0.0
def linear_penalty(overdue_time): return overdue_time
def quadtratic_penalty(overdue_time): return overdue_time**2

class TaskInstance:
    def __init__(
        self,
        task_category,
        task_id,
        instance_id,
        creation_time,
        duration_time,
        deadline_time,
        base_reward,
        penalty_fn
    ):
        self.task_category  = task_category
        self.task_id        = task_id
        self.instance_id    = instance_id
        self.creation_time  = creation_time
        self.duration_time  = duration_time
        self.deadline_time  = deadline_time
        self.base_reward    = base_reward
        self.penalty_fn     = penalty_fn

    def get_reward(self, current_time):
        overdue_time = max(0, current_time - self.deadline_time)
        return max(0, self.base_reward - self.penalty_fn(overdue_time))
        

    @property
    def label(self): return self.task_category.name

    def __repr__(self):
        return f"TaskInstance(Category={self.label}, id={self.instance_id}, duration={self.duration_time}, window=[{self.creation_time:.2f},{self.deadline_time:.2f}], reward={self.base_reward:.2f})"

class TaskCategory:
    def __init__(
        self,
        name,
        task_id, 
        category_seed,
        mean_time,
        std_time,
        mean_buffer_time,
        std_buffer_time,
        mean_reward,
        std_reward,
        penalty_fn
    ):
        self.name = name
        self.task_id = task_id
        self.category_seed = category_seed
        self.mean_time = mean_time
        self.std_time = std_time
        self.mean_buffer_time = mean_buffer_time
        self.std_buffer_time = std_buffer_time
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.penalty_fn = penalty_fn if penalty_fn else linear_penalty
        self._instance_count = 0

    def _rng(self, instance_id):
        seq = np.random.SeedSequence([self.category_seed, instance_id])
        return np.random.default_rng(seq.generate_state(1)[0])
    
    def create_instance(
            self,
            creation_time,
            instance_id
    ):
        if instance_id is None:
            instance_id = self._instance_count
            self._instance_count += 1
        
        rng = self._rng(instance_id)
        duration_time = max(1.0, rng.normal(self.mean_time, self.std_time))
        buffer_time = max(0.0, rng.normal(self.mean_buffer_time, self.std_buffer_time))
        deadline_time = duration_time + buffer_time
        base_reward = max(0.0, rng.normal(self.mean_reward, self.std_reward))
        return TaskInstance(
            task_category=self,
            task_id=self.task_id,
            instance_id=instance_id,
            creation_time=creation_time,
            duration_time=duration_time,
            deadline_time=deadline_time,
            base_reward=base_reward,
            penalty_fn=self.penalty_fn
        )

    def reset(self): self._instance_count = 0

    def __repr__(self):
        return f"TaskCategory(name={self.name},time~Normal({self.mean_time},{self.std_time}),reward~Normal({self.mean_reward},{self.std_reward}))"
    

class TaskGenerator:
    def __init__(
        self,
        task_category,
        generator_seed,
        probability
    ):
        self.task_category = task_category
        self.generator_seed = generator_seed
        self.probability = probability
        self._call_count = 0
        self._instance_count = 0

    def _rng(self, call_id):
        seq = np.random.SeedSequence([self.generator_seed, call_id])
        return np.random.default_rng(seq.generate_state(1)[0])

    def reset(self):
        self._call_count = 0
        self._instance_count = 0
        self.task_category.reset()

    def create_instance(
        self,
        creation_time,
        call_id=None
    ):
        if call_id is None:
            call_id = self._call_count
            self._call_count += 1
        rng = self._rng(call_id)
        if rng.uniform() > self.probability: return None

        instance = self.task_category.create_instance(
            creation_time=creation_time,
            instance_id=self._instance_count
        )
        self._instance_count += 1
        return instance
    
    def __repr__(self):
        return f"TaskGenerator(category={self.task_category.name}, p={self.probability})"