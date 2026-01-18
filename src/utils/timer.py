import time

class Timer:
    def __init__(self):
        self.time_list = []
        self.total_time = []

    def record(self):
        self.time_list.append(time.time())
    
    def infer_eta_time(self, cur_iter, max_iter):
        interval = (self.time_list[-1] - self.time_list[0]) / (len(self.time_list) - 1)

        rest_time = interval * (max_iter - 1 - cur_iter)
        rest_time = int(rest_time)

        t_m, t_s = divmod(rest_time, 60)
        t_h, t_m = divmod(t_m, 60)

        time_info = {
            'hours': t_h,
            'minutes': t_m,
            'seconds': t_s
        }

        return time_info
    