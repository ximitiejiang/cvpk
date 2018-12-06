from .hook import Hook

class TextHook(Hook):
    """输出文本显示: 可设置多少间隔显示一次iter"""
   
    def __init__(self, text_config):
        self.interval = text_config['interval']
        
    def before_train_epoch(self, runner):
        print('Epoch({}), [{}/{}]'.
              format(runner._epoch, runner._epoch+1, runner.cfg.epoch_num))
        
    def before_train_iter(self, runner):
        if self.interval > 0 and runner._iter % self.interval == 0:
            print('Iter({})'.format(runner._iter))
            
    def after_run(self, runner):
        print('Calculating Finish!')
