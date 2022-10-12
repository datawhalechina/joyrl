from common.utils import save_args,save_results,plot_rewards,timing
import time
class Launcher:
    def __init__(self) -> None:
        pass
    def get_args(self):
        cfg = {}
        return cfg
    def print_args(self,cfg):
        print("Hyperparameters:")
        print(''.join(['=']*80))
        tplt = "{:^20}\t{:^20}\t{:^20}"
        print(tplt.format("Name", "Value", "Type"))
        for k,v in cfg.items():
            print(tplt.format(k,v,str(type(v))))   
        print(''.join(['=']*80))
    def env_agent_config(self,cfg):
        env,agent = None,None
        return env,agent
    def train(self,cfg, env, agent):
        res_dic = {}
        return res_dic
    def test(self,cfg, env, agent):
        res_dic = {}
        return res_dic
    def run(self):
        cfg = self.get_args()
        self.print_args(cfg)
        env, agent = self.env_agent_config(cfg)
        start_time = time.time()
        res_dic = self.train(cfg, env, agent)
        end_time = time.time()
        training_time = end_time - start_time
        cfg.update({"training_time":training_time}) # update to cfg paramters
        save_args(cfg,path = cfg['result_path']) # save parameters
        agent.save_model(path = cfg['model_path'])  # save models
        save_results(res_dic, tag = 'train', path = cfg['result_path']) # save results
        plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "train")  # plot results
        # testing
        # env, agent = self.env_agent_config(cfg) # create new env for testing, sometimes can ignore this step
        agent.load_model(path = cfg['model_path'])  # load model
        res_dic = self.test(cfg, env, agent)
        save_results(res_dic, tag='test',
                    path = cfg['result_path'])  
        plot_rewards(res_dic['rewards'], cfg, path = cfg['result_path'],tag = "test") 
