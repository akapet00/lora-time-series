import datetime

class Timer():
    def __init__(self):
        self.start_dt = None 

    def start(self):
        self.start_dt = datetime.datetime.now()

    def stop(self):
        end_dt = datetime.datetime.now()
        print('Time: %s' % (end_dt - self.start_dt))