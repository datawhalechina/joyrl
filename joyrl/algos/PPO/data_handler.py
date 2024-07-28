#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-17 01:08:36
LastEditor: JiangJi
LastEditTime: 2024-06-23 17:51:33
Discription: 
'''
import numpy as np
from joyrl.algos.base.data_handler import BaseDataHandler

class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gae_lambda = getattr(self.cfg, 'gae_lambda', 0.95)
        self.gamma = getattr(self.cfg, 'gamma', 0.95)
        self.batch_exps = []
        
    def handle_exps_after_interact(self, exps):
        exp_len = self._get_exp_len(exps)
        next_value = exps[-1].value
        return_mc = 0
        return_td = next_value   
        adv_gae = 0
        returns_mc = []
        returns_td = []
        returns_gae = []
        for t in reversed(range(exp_len)):
            delta = exps[t].reward + self.gamma * next_value * (1 - exps[t].done) - exps[t].value
            adv_gae = delta + self.gamma * self.gae_lambda * (1 - exps[t].done) * adv_gae
            return_mc = exps[t].reward + self.gamma * return_mc * (1 - exps[t].done)
            return_td = exps[t].reward + self.gamma * return_td * (1 - exps[t].done)
            returns_mc.insert(0, return_mc)
            returns_td.insert(0, return_td)
            returns_gae.insert(0, adv_gae + exps[t].value)
            exps[t].return_mc = return_mc
            exps[t].return_td = return_td
            exps[t].adv_gae = adv_gae
            exps[t].return_gae = adv_gae + exps[t].value
            next_value = exps[t].value
        return_mc_normed = (returns_mc - np.mean(returns_mc)) / (np.std(returns_mc) + 1e-8)
        return_td_normed = (returns_td - np.mean(returns_td)) / (np.std(returns_td) + 1e-8)
        return_gae_normed = (returns_gae - np.mean(returns_gae)) / (np.std(returns_gae) + 1e-8)
        for t in range(exp_len):
            exps[t].return_mc_normed = return_mc_normed[t]
            exps[t].normed_return_td = return_td_normed[t]
            exps[t].normed_return_gae = return_gae_normed[t]
        exps = exps[:exp_len]
        return exps
    
    def add_exps(self, exps):
        exps = self.handle_exps_after_interact(exps)
        self.buffer.push(exps)

    def _handle_exps_before_train(self, exps: list):
        ''' convert exps to training data
        '''
        super()._handle_exps_before_train(exps)
        log_probs = [exp.log_prob for exp in exps] 
        # log_probs = torch.cat(log_probs, dim=0).detach() # [batch_size,1]
        # log_probs = torch.tensor(log_probs, dtype = torch.float32, device = self.cfg.device).unsqueeze(dim=1)
        if self.cfg.return_form.lower() == 'mc':
            returns = np.array([exp.return_mc_normed for exp in exps])
        elif self.cfg.return_form.lower() == 'td':
            returns = np.array([exp.normed_return_td for exp in exps])
        elif self.cfg.return_form.lower() == 'gae':
            returns = np.array([exp.normed_return_gae for exp in exps])
        else:
            raise NotImplementedError("return_form not implemented")
        # returns = torch.tensor(returns, dtype = torch.float32, device = self.cfg.device).unsqueeze(dim=1)
        adv_gae_np = np.array([exp.adv_gae for exp in exps])
        self.data_after_train.update({
            'log_probs': log_probs, 
            'returns': returns,
            'return_gae': np.array([exp.return_gae for exp in exps]),
            'adv': (adv_gae_np - np.mean(adv_gae_np)) / (np.std(adv_gae_np) + 1e-8),
            'before_v': np.array([exp.value for exp in exps]),
        })

        