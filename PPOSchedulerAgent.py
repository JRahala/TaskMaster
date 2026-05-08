import torch
from utils import Log


class PPOSchedulerAgent:
    name="Scheduler"
    def __init__(self, model, horizon):
        self.H = horizon
        self.model = model
        self.schedule_window = [-1] * self.H
        self.job_ids = {}
        self.t = 0
        self.logger = Log()

    def plan(self, available_jobs, timestep):

        for job in available_jobs:
            if job.instance_id not in self.job_ids.keys():
                self.insert_task(job)
        # self.shift()
        schedule = self.get_schedule_window()
        schedule.shift()
        return schedule

    def reset(self):
        self.schedule_window = [-1] * self.H
        self.job_ids = {}
        self.t = 0

    def get_schedule_window(self):
        return self.schedule_window

    def embed_schedule(self):
        schedule_embedding = [[-1, 0] if x == -1 else [self.job_ids[x][0].task_id, 
                                                    self.job_ids[x][0].deadline_time - self.t] for x in self.schedule_window]
        return schedule_embedding
    
    def insert_task(self, job):
        # change to encode based on types
        schedule_tensor = torch.tensor(self.embed_schedule())
        job_tensor = torch.tensor([job.task_id, job.deadline_time - self.t] if job else [-1, 0])
        pred_length = self.model.get_pred_length(job_tensor)
        mask_tensor = torch.tensor(self.valid_mask(pred_length))

        action, _ = self.model.get_action(schedule_tensor, job_tensor, mask_tensor)
        action_item = action.item()

        self.place(job, pred_length, action_item)

    def check_reward(self, job_id):
        job, pred_length, progress = self.job_ids[job_id]
        if progress == job.duration_time:
            return job.get_reward(self.t)
        elif progress > pred_length and progress < job.duration_time:
            return -1
        return 0
    
    def free_future_instances(self, job_id):
        for i in range(self.H):
            if self.schedule_window[i] == job_id:
                self.schedule_window[i] = -1

    def encode_job(self, job_id):
        job = self.job_ids[job_id][0]
        job_vals = [job.task_id, job.deadline_time - self.t] if job else [-1, 0]
        return job_vals

    def valid_mask(self, pred_length):
        mask = []
        for i in range(self.H):
            if i + pred_length > self.H:
                mask.append(0)
            else:
                ok = all(self.schedule_window[i + k] == -1 for k in range(pred_length))
                mask.append(1 if ok else 0)
        return mask

    def can_place(self, i, pred_length):
        if i + pred_length > self.H:
            return False
        return all(self.schedule_window[i + k] == -1 for k in range(pred_length))

    def place(self, job, pred_length, i):
        if not job.instance_id in self.job_ids:
            self.job_ids[job.instance_id] = [job, pred_length, 0]
        for k in range(pred_length):
            self.schedule_window[i + k] = job.instance_id

    def shift(self):
        job = self.schedule_window[0]
        if job != -1:
            self.job_ids[job][2] += 1
        self.schedule_window = self.schedule_window[1:] + [-1]
        self.t += 1
        return job
    
    def copy(self):
        s = PPOSchedulerAgent(self.model, self.H)
        s.schedule_window = self.schedule_window.copy()
        s.job_ids = self.job_ids.copy()
        s.t = self.t
        return s