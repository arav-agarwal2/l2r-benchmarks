import itertools
from copy import deepcopy

import torch
import numpy as np
from gym.spaces import Box
from torch.optim import Adam

from src.agents.base import BaseAgent
from src.config.yamlize import yamlize, create_configurable, NameToSourcePath
from src.utils.utils import ActionSample

from src.constants import DEVICE


@yamlize
class PCPOAgent(BaseAgent):
    """Adopted from https://github.com/PKU-MARL/Safe-Policy-Optimization"""

    def __init__(
        self,
        steps_to_sample_randomly: int,
        gamma: float,
        alpha: float,
        cost_limit: float,
        target_kl: float,
        cg_damping: float,
        polyak: float,
        lr: float,
        actor_critic_cfg_path: str,
        load_checkpoint_from: str = "",
    ):
        """Initialize Soft Actor-Critic Agent

        Args:
            steps_to_sample_randomly (int): Number of steps to sample randomly
            gamma (float): Gamma parameter
            alpha (float): Alpha parameter
            polyak (float): Polyak parameter coef.
            lr (float): Learning rate parameter.
            actor_critic_cfg_path (str): Actor Critic Config Path
            load_checkpoint_from (str, optional): Load checkpoint from path. If '', then doesn't load anything. Defaults to ''.
        """

        super(PCPOAgent, self).__init__()

        self.steps_to_sample_randomly = steps_to_sample_randomly
        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.load_checkpoint_from = load_checkpoint_from
        self.lr = lr
        self.cost_limit = cost_limit
        self.t = 0
        self.target_kl = target_kl
        self.loss_pi_cost_before = 0.0
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0
        self.deterministic = False
        self.cg_damping = cg_damping

        self.record = {"transition_actor": ""}  # rename
        self.use_cost_value_function = True
        self.action_space = Box(-1, 1, (2,))
        self.act_dim = self.action_space.shape[0]
        self.obs_dim = 32

        self.actor_critic = create_configurable(
            actor_critic_cfg_path, NameToSourcePath.network
        )
        self.actor_critic.to(DEVICE)
        self.actor_critic_target = deepcopy(self.actor_critic)

        if self.load_checkpoint_from != "":
            self.load_model(self.load_checkpoint_from)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.policy.parameters(), lr=self.lr)
        self.vf_optimizer = Adam(self.actor_critic.v.parameters(), lr=self.lr)
        self.cf_optimizer = Adam(self.actor_critic.c.parameters(), lr=self.lr)
        
        self.pi_scheduler = (
            torch.optim.lr_scheduler.StepLR(  # TODO: Call some scheduler in runner.
                self.pi_optimizer, 1, gamma=0.5
            )
        )

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False

    def get_flat_params_from(model):
        flat_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                d = param.data
                d = d.view(-1)  # flatten tensor
                flat_params.append(d)
        assert flat_params is not [], 'No gradients were found in model parameters.'
    
    def set_param_values_to_model(model, vals):
        assert isinstance(vals, torch.Tensor)
        i = 0
        for name, param in model.named_parameters():
            if param.requires_grad:  # param has grad and, hence, must be set
                orig_size = param.size()
                size = np.prod(list(param.size()))
                new_values = vals[i:i + size]
                # set new param values
                new_values = new_values.view(orig_size)
                param.data = new_values
                i += size  # increment array position
        assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'

    def adjust_cpo_step_direction(
            self,
            step_dir,
            g_flat,
            optim_case,
            p_dist,
            data,
            total_steps: int = 25,
            decay: float = 0.8
    ):
        """
            PCPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs.
        """
        step_frac = 1.0
        _theta_old = self.get_flat_params_from(self.actor_critic.policy.net)
        _, old_log_p = self.actor_critic.pi(data['obs']) 
        expected_rew_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            self.set_param_values_to_model(self.actor_critic.policy.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi_rew, _ = self.compute_loss_pi(data=data)
                loss_pi_cost, _ = self.compute_loss_cost_performance(data=data)
                # determine KL div between new and old policy
                q_dist = self.actor_critic.policy.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            loss_rew_improve = self.loss_pi_before - loss_pi_rew.item()
            cost_diff = loss_pi_cost.item() - self.loss_pi_cost_before

            if not torch.isfinite(loss_pi_rew) and not torch.isfinite(
                    loss_pi_cost):
                print('WARNING: loss_pi not finite')
            elif loss_rew_improve < 0 if optim_case > 1 else False:
                print('INFO: did not improve improve <0')

            elif torch_kl > self.target_kl * 1.5:
                print(
                    f'INFO: violated KL constraint {torch_kl} at step {j + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                print(f'Accept step at i={j + 1}')
                break
            step_frac *= decay
        else:
            print('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        self.set_param_values_to_model(self.actor_critic.policy.net, _theta_old)
        return step_frac * step_dir, acceptance_step

    
    def select_action(self, obs):
        """Select action from obs.

        Args:
            obs (np.array): Observation to act on.

        Returns:
            ActionObj: Action object.
        """
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        action_obj = ActionSample()
        if self.t > self.steps_to_sample_randomly:
            a, v, c, logp_a = self.actor_critic.step(obs.to(DEVICE), self.deterministic)
            a = a  # numpy array...
            action_obj.action = a
            action_obj.value = v
            action_obj.cost = c
            action_obj.logp = logp_a
            self.record["transition_actor"] = "learner"
        else:
            a = self.action_space.sample()
            logp = np.ones((1,))
            v = np.ones((1,))
            c = np.ones((1,))
            action_obj.action = a
            action_obj.value = v
            action_obj.cost = c
            action_obj.logp = logp
            self.record["transition_actor"] = "random"
        self.t = self.t + 1
        return action_obj

    def register_reset(self, obs):
        """
        Same input/output as select_action, except this method is called at episodal reset.
        """
        pass

    def load_model(self, path):
        """Load model from path.

        Args:
            path (str): Load model from path.
        """
        self.actor_critic.load_state_dict(torch.load(path))

    def save_model(self, path):
        """Save model to path

        Args:
            path (str): Save model to path
        """
        torch.save(self.actor_critic.state_dict(), path)

    def compute_loss_cost_performance(self, data):
        dist, _log_p = self.actor_critic.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        cost_loss = (ratio * data['cost_adv']).mean()
        # ent = dist.entropy().mean().item()
        info = {}
        return cost_loss, info


    def compute_loss_v(self, obs, ret):
        """
        computing value loss
        Returns:
            torch.Tensor
        """
        return ((self.actor_critic.v(obs) - ret) ** 2).mean()

    def compute_loss_pi(self, data):
        """Set up function for computing SAC pi loss."""
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def compute_loss_c(self, obs, ret):
        """
        computing cost loss
        Returns:
            torch.Tensor
        """
        return ((self.actor_critic.c(obs) - ret) ** 2).mean()


    def update(self):
        """
            Update actor, critic, running statistics
        """
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Update Policy Network
        self.update_policy_net(data)
        # Update Value Function
        self.update_value_net(data=data)
        if self.use_cost_value_function:
            self.update_cost_net(data=data)
    
    def update_value_net(self, data: dict) -> None:
        # Divide whole local epoch data into mini_batches which is mbs size

        loss_v = self.compute_loss_v(data['obs'], data['target_v'])
        self.loss_v_before = loss_v.item()
        self.vf_optimizer.zero_grad()
        loss_v = self.compute_loss_v(
            obs=data['obs'],
            ret=data['target_v'])
        loss_v.backward()
        self.vf_optimizer.step()

    def update_cost_net(self, data: dict) -> None:
        # Ensure we have some key components
        assert self.use_cost_value_function
        assert hasattr(self, 'cf_optimizer')
        assert 'target_c' in data, f'provided keys: {data.keys()}'

        if self.use_cost_value_function:
            self.loss_c_before = self.compute_loss_c(data['obs'],
                                                     data['target_c']).item()

        # Train cost value network
        self.cf_optimizer.zero_grad()
        loss_c = self.compute_loss_c(obs=data['obs'],
                                        ret=data['target_c'])
        loss_c.backward()
        self.cf_optimizer.step()
    
    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = self.get_flat_params_from(self.ac.pi.net)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        self.loss_v_before = self.compute_loss_v(data['obs'],
                                                 data['target_v']).item()
        self.loss_c_before = self.compute_loss_c(data['obs'],
                                                 data['target_c']).item()
        # get prob. distribution before updates
        p_dist = self.actor_critic.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        
        g_flat = self.get_flat_gradients_from(self.actor_critic.pi.net)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = self.conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        eps = 1.0e-8
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        H_inv_g = self.Fvp(x)
        alpha = torch.sqrt(2 * self.target_kl / (xHx + eps))
        assert xHx.item() >= 0, 'No negative values'

        # get the policy cost performance gradient b (flat as vector)
        self.pi_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(data=data)
        loss_cost.backward()
        
        self.loss_pi_cost_before = loss_cost.item()
        b_flat = self.get_flat_gradients_from(self.actor_critic.pi.net)


        # set variable names as used in the paper
        p = self.conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        q = xHx
        r = g_flat.dot(p)  # g^T H^{-1} b
        s = b_flat.dot(p)  # b^T H^{-1} b
        
        step_dir = torch.sqrt(2 * self.target_kl / (q + 1e-8)) * H_inv_g - torch.clamp_min((torch.sqrt(2 * self.target_kl/q) * r + c)/ s, torch.tensor(0.0)) * p

        final_step_dir, accept_step = self.adjust_cpo_step_direction(
            step_dir,
            g_flat,
            optim_case=2,
            p_dist=p_dist,
            data=data,
            total_steps=20
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        self.set_param_values_to_model(self.actor_critic.pi.net, new_theta)

        q_dist = self.ac.pi.dist(data['obs'])
        torch_kl = torch.distributions.kl.kl_divergence(
            p_dist, q_dist).mean().item()


    def conjugate_gradients(self, Avp, b, nsteps, residual_tol=1e-10, eps=1e-6):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        nsteps: (int): Number of iterations of conjugate gradient to perform.
                Increasing this will lead to a more accurate approximation
                to :math:`H^{-1} g`, and possibly slightly-improved performance,
                but at the cost of slowing things down.
                Also probably don't play with this hyperparameter.
        """
        x = torch.zeros_like(b)
        r = b - Avp(x)
        p = r.clone()
        rdotr = torch.dot(r, r)

        fmtstr = "%10i %10.3g %10.3g"
        titlestr = "%10s %10s %10s"
        verbose = False

        for i in range(nsteps):
            if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
            z = Avp(p)
            alpha = rdotr / (torch.dot(p, z) + eps)
            x += alpha * p
            r -= alpha * z
            new_rdotr = torch.dot(r, r)
            if torch.sqrt(new_rdotr) < residual_tol:
                break
            mu = new_rdotr / (rdotr + eps)
            p = r + mu * p
            rdotr = new_rdotr

        return x

    def get_flat_gradients_from(self, model):
        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad
                grads.append(g.view(-1))  # flatten tensor and append
        assert grads is not [], 'No gradients were found in model parameters.'

        return torch.cat(grads)
    
    def Fvp(self, p):
        """ 
            Build the Hessian-vector product based on an approximation of the KL-divergence.
            For details see John Schulman's PhD thesis (pp. 40) http://joschu.net/docs/thesis.pdf
        """
        self.ac.pi.net.zero_grad()
        q_dist = self.actor_critic.pi.dist(self.fvp_obs)
        with torch.no_grad():
            p_dist = self.actor_critic.pi.dist(self.fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.actor_critic.pi.net.parameters(),
                                    create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * p).sum()
        grads = torch.autograd.grad(kl_p, self.ac.pi.net.parameters(),
                                    retain_graph=False)
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                       for grad in grads])
        
        return flat_grad_grad_kl + p * self.cg_damping