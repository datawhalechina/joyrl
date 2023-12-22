import ray
from ray.util.queue import Queue as RayQueue
import multiprocessing as mp
from queue import Queue
import threading
from joyrl.framework.message import Msg, MsgType
from joyrl.framework.config import MergedConfig
from joyrl.algos.base.data_handler import BaseDataHandler
from joyrl.framework.base import Moduler

class Collector(Moduler):
    ''' Collector for collecting training data
    '''
    def __init__(self, cfg: MergedConfig, *args, **kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.data_handler =kwargs['data_handler']
        self.logger = kwargs['logger']
        self.training_data_que = Queue(maxsize = self.cfg.n_learners + 1) if not ray.is_initialized() else RayQueue(maxsize = self.cfg.n_learners + 1)

    def _t_start(self):
        self._t_sample_training_data = threading.Thread(target=self._sample_training_data)
        self._t_sample_training_data.setDaemon(True)
        self._t_sample_training_data.start()
        
    def _p_start(self):
        self._p_sample_training_data = mp.Process(target=self._sample_training_data)
        self._p_sample_training_data.start()
    
    def init(self):
        if self.use_ray:
            self.logger.info.remote("[Collector.init] Start collector!")
        else:
            self.logger.info("[Collector.init] Start collector!")
            # self._p_start()
        self._t_start()

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps = msg_data
            self._put_exps(exps)
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            if self.training_data_que.empty(): return None
            try:
                return self.training_data_que.get(block = False)
            except:
                return None
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError

    def _sample_training_data(self):
        ''' 
        '''
        while True:
            training_data = self._get_training_data()
            if training_data is not None:
                try:
                    self.training_data_que.put(training_data, block = False)
                except:
                    pass
                    # print("training_data_que is full!")
                    # import time
                    # time.sleep(0.001)
            
    def _put_exps(self, exps):
        ''' add exps to data handler
        '''
        self.data_handler.add_exps(exps) # add exps to data handler

    def _get_training_data(self):
        training_data = self.data_handler.sample_training_data() # sample training data
        return training_data
    
    def handle_data_after_learn(self, policy_data_after_learn, *args, **kwargs):
        return 
    
    def get_buffer_length(self):
        return len(self.data_handler.buffer)
