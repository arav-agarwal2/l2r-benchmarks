"""Utility functions used throughout the source. Should generally not add here unless it helps significantly."""
from dataclasses import dataclass
import os, sys
import logging, re
from termios import VQUIT
import numpy as np
from datetime import datetime
import torch
from scipy import stats

def find_envvar_patterns(self, config, key):
    pattern = re.compile(".*?\${(\w+)}.*?")
    try:
        envvars = re.findall(pattern, config[key])
    except:
        envvars = []
        pass
    return envvars


def replace_envvar_patterns(self, config, key, envvars, args):
    for i, var in enumerate(envvars):
        if var == "DIRHASH":
            dirhash = "{}/".format(args.dirhash) if not args.runtime == "local" else ""
            config[key] = config[key].replace("${" + var + "}", dirhash)
        if var == "PREFIX":
            prefix = {"local": "/data", "phoebe": "/mnt"}
            config[key] = config[key].replace("${" + var + "}", prefix[args.runtime])
        else:
            config[key] = config[key].replace(
                "${" + var + "}", os.environ.get(var, var)
            )


def resolve_envvars(self, config, args):

    for key in list(config.keys()):

        if isinstance(config[key], dict):
            # second level
            for sub_key in list(config[key].keys()):
                sub_envvars = find_envvar_patterns(config[key], sub_key)
                if len(sub_envvars) > 0:
                    for sub_var in sub_envvars:
                        replace_envvar_patterns(config[key], sub_key, sub_envvars, args)

        envvars = find_envvar_patterns(config, key)
        if len(envvars) > 0:
            replace_envvar_patterns(config, key, envvars, args)

    return config


def is_number(self, s):
    """
    Somehow, the most pythonic way to check string for float number; used for safe user input parsing
    src: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


class RecordExperience:
    def __init__(self, record_dir, track, experiment_name, logger, agent=False):

        self.record_dir = record_dir
        self.track = track
        self.experiment_name = experiment_name
        self.filename = "transition"
        self.agent = agent
        self.logger = logger

        self.path = os.path.join(self.record_dir, self.track, self.experiment_name)

        self.logger("Recording agent experience")

    def save(self, record):

        filename = f"{self.path}/{record['stage']}/{record['episode']}/{self.filename}_{self.experiment_name}_{record['step']}"

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(
            os.path.join(self.path, record["stage"], str(record["episode"])),
            exist_ok=True,
        )

        np.savez_compressed(filename, **record)

        return record

    def save_thread(self):
        """Meant to be run as a separate thread"""
        if not self.agent:
            raise Exception("RecordExperience requires an SACAgent")

        while True:
            batch = self.agent.save_queue.get()
            self.logger("[RecordExperience] Saving experience.")
            for record in batch:
                self.save(record)

# Ref https://andyljones.com/posts/rl-debugging.html for ideal values and meaning of metrics
# https://github.com/andyljones/boardlaw/tree/master is a good reference also.
# Relatively untested and unintegrated with the rest of the codebase
class DebuggingRL:
    # If it stays close to 1 then you're not learning, 
    # if it is close to 0, you've learned some policy that's now stable and is no longer exploring
    # If it is oscillating a lot then maybe reduce the LR. 
    def __init__(self, plot_after_steps = 1000, residual_variance_after_steps = 1000, plot_ep_lengths_after=100):
        self.rewards = list()
        self.value_targets = list()
        self.values = list()
        self.episode_lengths = list()
        self.value_targets_var = list()
        self.net_values = list()
        self.plot_after_steps = plot_after_steps # Unit is steps
        self.plot_ep_lengths_after = plot_ep_lengths_after # Unit is episodes 
        self.residual_variance_after_steps = residual_variance_after_steps
        self.step_counter = 0
        self.ep_counter = 0
        self.residual_variance_counter = 0
        self.reward_step_counter = 0

    # May need to be fixed - runs without error but value not bounded in [0,1]
    def relative_policy_entropy(self, log_prob):
        valid = (log_prob > -np.inf)
        zeros = torch.zeros_like(log_prob)
        log_prob = log_prob.where(valid, zeros)
        probs = log_prob.exp().where(valid, zeros)
        rel_polent = (-(log_prob*probs).sum(-1).mean())/(torch.log(valid.sum(-1).float()).mean())
        return rel_polent

    # Assumes Gaussian
    # Follows formula mentioned here: https://stats.stackexchange.com/a/7449
    def KLdivergence(self, old_policy, new_policy):
        # policies are tuples of form (mu, logvar) which is what the policy net returns currently
        old_mu = old_policy[0]
        old_log_std =  old_policy[1].repeat(old_mu.shape[1],1).T
        new_mu = new_policy[0]
        new_log_std =  new_policy[1].repeat(new_mu.shape[1],1).T
        kl_div = (new_log_std - old_log_std) + (torch.exp(old_log_std)**2 + (old_mu - new_mu)**2)/(2*torch.exp(new_log_std)**2) - 0.5
        return kl_div.mean(axis=0)

    # When called, log if returned value is not None
    def residual_variance(self, target_val, net_val):
        self.residual_variance_counter+=1
        self.value_targets_var.append(target_val.mean().item())
        self.net_values.append(net_val.mean().item())
        if(self.residual_variance_counter >= self.residual_variance_after_steps):
            net_val_arr = np.array(self.net_values)
            targ_val_arr = np.array(self.value_targets_var)
            self.residual_variance_counter = 0
            self.value_targets_var = list()
            self.net_values = list()
            #print((np.std(targ_val_arr - net_val_arr)**2)/(np.std(targ_val_arr)**2))
            return (np.std(targ_val_arr - net_val_arr)**2)/(np.std(targ_val_arr)**2)
        return None 

    # Not implemented since large state space. Ref https://andyljones.com/posts/rl-debugging.html for why this might be useful
    def reward_state_correlation():
        raise NotImplementedError()

    def get_summary_stats(self, vallist, valtype="Reward"):
        print(f"For {valtype}")
        print(stats.describe(vallist))

    def __get_stats(self):
        if(len(self.rewards) > 0):
            self.get_summary_stats(self.rewards, valtype="Reward")
        if(len(self.values) > 0):
            self.get_summary_stats(self.values, valtype="Values")
        if(len(self.value_targets) > 0):
            self.get_summary_stats(self.value_targets, valtype="Target Values")

    # default None so you don't have to log all of them
    def collect_values_value_targets(self, value=None, value_target=None):
        self.step_counter+=1
        if(value is not None):
            self.values.append(value.mean().item())
        if(value_target is not None):
            self.value_targets.append(value_target.mean().item())
        
        if(self.step_counter >= self.plot_after_steps):
            self.__get_stats()
            self.step_counter = 0
            self.values = list()
            self.value_targets = list()
    
    def collect_rewards(self, reward=None):
        self.reward_step_counter+=1
        if(reward is not None):
            self.rewards.append(reward)
        
        if(self.reward_step_counter >= self.plot_after_steps):
            self.__get_stats()
            self.reward_step_counter = 0
            self.rewards = list()

    def collect_episode_lengths(self, ep_len):
        self.ep_counter+=1
        self.episode_lengths.append(ep_len)
        if(self.ep_counter >= self.plot_ep_lengths_after):
            self.get_summary_stats(self.episode_lengths, valtype="Episode length")
            self.ep_counter = 0
            self.episode_lengths = list()
    
    # Requires slight modification of network, not implemented for now
    def sample_staleness(self):
        raise NotImplementedError()
    

    def step_stats(self, old_net_params, new_net_params, nettype="Policy"):
        param_diffs = list()
        for old_net_param, new_net_param in zip(old_net_params, new_net_params):
            param_diff = new_net_param.data - old_net_param.data
            param_diffs.append(param_diff)

        abs_maxs = list()
        mses = list()
        for param_diff in param_diffs:
            abs_maxs.append(torch.max(torch.abs(param_diff)).item())
            mses.append(torch.mean(torch.square(param_diff)).item())
        return abs_maxs, mses

    # We are currently using Adam so this is omitted 
    def gradient_stats(self):
        raise NotImplementedError()

    # Not implemented since we have used more rudimentary methods of measuring this
    # Can implement in future if needed
    def component_throughput(self):
        raise NotImplementedError()



@dataclass
class ActionSample:
    """Generic object to store action-related params. Might be useful to remove."""

    action = None
    value = None
    logp = None
