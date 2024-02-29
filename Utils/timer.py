import time
import numpy as np

class Timer:
    def __init__(self):
        """The initialize of one timer"""
        self.times = []
        self.start()
        self.timer_working = False
        self.tik = 0
    def start(self):
        """Start the Timer"""
        if not self.timer_working:
            self.tik = time.time()
            self.timer_working = True
    def stop(self):
        """pause the timer and record the time"""
        if self.timer_working:
            self.times.append(time.time() - self.tik)
            return self.times[-1]
        else:
            return -1 
    def avg_time(self):
        """Return the average value of """
        return sum(self.times) /  len(self.times)

    def sum_time(self):
        """Return the sum of times"""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulating time"""
        return np.array(self.times).cumsum().tolist()

    


