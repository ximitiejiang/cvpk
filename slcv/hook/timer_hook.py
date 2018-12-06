import time

from .hook import Hook


class TimerHook(Hook):
    """记录各iter/epoch的运行时间: 可以在任何位置调用log()打印输出
    平均iter和平均epoch时间
    """
    def __init__(self):
        self.epoch_time = []
        self.avg_epoch_time = 0
    
    def log(self,runner):
        runner.log_buffer.average(0)  # 0代表求所有buffer中数据的平均值
        if runner.log_buffer.ready:
            avg_iter_time = runner.log_buffer.average_output['iter_time']
        avg_epoch_time =  sum(self.epoch_time)/len(self.epoch_time)
        print('Avg iter time: {}, avg epoch time: {}'.
              format(avg_iter_time, avg_epoch_time))
        print('\nTotal time: {}'.format(sum(self.epoch_time)))
    
    def before_train_epoch(self, runner):
        self.epoch_st = time.time()
        
    def before_train_iter(self, runner):
        self.iter_st = time.time()
#        runner.log_buffer.update({'data_time': time.time() - self.epoch_time})

    def after_train_iter(self, runner):
        runner.log_buffer.update({'iter_time': time.time() - self.iter_st})
        self.iter_st = time.time()
    
    def after_train_epoch(self, runner):
        self.epoch_time.append(time.time()-self.epoch_st)
        self.epoch_st = time.time()
    
    def after_run(self, runner):
        self.log(runner)