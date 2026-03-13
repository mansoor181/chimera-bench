
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abflownet.modules.common.layers import clampped_one_hot
from abflownet.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class PositionTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(p_t|p_0)
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_rand = torch.randn_like(p_0)
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)

        if return_prob:
            diff = p_noisy - c0*p_0
            var = c1**2
            # log p_F(p_t|p_0)
            log_prob = -0.5*torch.sum(diff**2/var, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log(c1.squeeze(-1))
            log_prob = torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))
            return p_noisy, e_rand, log_prob
        else:
            return p_noisy, e_rand

    def add_noise_step_wise(self, p_t_1, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(p_t|p_{t-1}) = N(p_t | \sqrt{1-β} · x_{t-1}, β*I)
        """
        beta = self.var_sched.betas[t]
        c0 = torch.sqrt(1-beta).view(-1, 1, 1)
        c1 = torch.sqrt(beta).view(-1, 1, 1)

        e_rand = torch.randn_like(p_t_1)
        p_noisy = c0*p_t_1 + c1*e_rand
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_t_1), p_noisy, p_t_1)

        if return_prob:
            diff = p_noisy - c0*p_t_1
            var = c1**2
            # log p_F(p_t|p_t_1)
            log_prob = -0.5*torch.sum(diff**2/var, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log(c1.squeeze(-1))
            log_prob = torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))
            #approx gaussian but not exact. log prob must be between -inf and 0
            log_prob = torch.clamp(log_prob, min=-1e6, max=0)
            
            return p_noisy, e_rand, log_prob
        else:
            return p_noisy, e_rand


    def denoise(self, p_t, eps_p, mask_generate, t, return_prob=False):
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(-1, 1, 1)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(-1, 1, 1)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)

        if return_prob:
            # Calculate mean of the posterior distribution
            mu = c0 * (p_t - c1 * eps_p)
            
            # Calculate log probability
            diff = p_next - mu
            var = sigma**2 + 1e-8
            # log p(p_{t-1}|p_t)
            log_prob = -0.5*torch.sum(diff**2/var, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log(sigma.squeeze(-1))
            log_prob = torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))
            #approx gaussian but not exact. log prob must be between -inf and 0
            log_prob = torch.clamp(log_prob, min=-1e6, max=0)
            
            return p_next, log_prob
        else:
            return p_next


class RotationTransition(nn.Module):
    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        beta = self.var_sched.betas
        self.angular_distrib_step_wise = ApproxAngularDistribution(beta.tolist(), **angular_distrib_fwd_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))

    def add_noise(self, v_0, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(v_t|v_0)
        """
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device)
        R0_scaled = so3vec_to_rotation(c0 * v_0)
        E_scaled = so3vec_to_rotation(e_scaled)
        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        if return_prob:
            diff = v_noisy - (c0 * v_0)
            var = c1**2
            # log p_F(v_t|v_0)
            log_prob = -0.5 * torch.sum(diff**2/var, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log(c1.squeeze(-1))
            log_prob = torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))
            return v_noisy, e_scaled, log_prob
        else:
            return v_noisy, e_scaled


    def add_noise_step_wise(self, v_t_1, mask_generate, t, return_prob=False):
        """
        Step-wise forward transition q(v_t|v_{t-1})
        """
        N, L = mask_generate.size()
        beta = self.var_sched.betas[t]
        c0 = torch.sqrt(1-beta).view(-1, 1, 1)
        c1 = torch.sqrt(beta).view(-1, 1, 1)

        e_scaled = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_step_wise, device=self._dummy.device)
        Rt_scaled = so3vec_to_rotation(c0 * v_t_1)
        E_scaled = so3vec_to_rotation(e_scaled)
        R_noisy = E_scaled @ Rt_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_t_1), v_noisy, v_t_1)

        if return_prob:
            diff = v_noisy - (c0 * v_t_1)
            var = c1**2
            # log q(v_t|v_{t-1})
            log_prob = -0.5 * torch.sum(diff**2/var, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log(c1.squeeze(-1))
            log_prob = torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))

            #approx gaussian but not exact. log prob must be between -inf and 0
            log_prob = torch.clamp(log_prob, min=-1e6, max=0)
            return v_noisy, e_scaled, log_prob
        else:
            return v_noisy, e_scaled


    def denoise(self, v_t, v_next, mask_generate, t):
        # Unchanged
        N, L = mask_generate.size()
        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e,
            torch.zeros_like(e)
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next

    def denoise(self, v_t, v_next, mask_generate, t, return_prob=False):
        """
        Denoising step with optional probability calculation
        Args:
            v_t: Current state
            v_next: Predicted next state
            mask_generate: Generation mask
            t: Time steps
            return_prob: Whether to return probability
        Returns:
            v_next: Denoised state
            e: Random noise
            log_prob: Log probability (if return_prob=True)
        """
        N, L = mask_generate.size()
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)
    
        e = random_normal_so3(t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e,
            torch.zeros_like(e)
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next_noisy = rotation_to_so3vec(R_next)
        v_next_noisy = torch.where(mask_generate[..., None].expand_as(v_next_noisy), v_next_noisy, v_t)

        if return_prob:
            diff = v_next_noisy - v_next
            var = sigma**2 + 1e-8
            # log p(v_{t-1}|v_t)
            log_prob = -0.5 * torch.sum(diff**2/var, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log(sigma.squeeze(-1))
            log_prob = torch.where(mask_generate, log_prob, torch.zeros_like(log_prob))
            #approx gaussian but not exact. log prob must be between -inf and 0
            log_prob = torch.clamp(log_prob, min=-1e6, max=0)
            
            return v_next_noisy, log_prob
        else:
            return v_next_noisy


    # def backward_prob(self, v_t, v_0, mask_generate, t):
    #     """
    #     Backward transition p_B(v_0|v_t)
    #     Similarly as positions:
    #     p_B(v_0|v_t) = N(v_0; v_t/c0, (c1^2/c0^2)*I)
    #     """
    #     alpha_bar = self.var_sched.alpha_bars[t]
    #     c0 = torch.sqrt(alpha_bar).view(-1,1,1)
    #     c1 = torch.sqrt(1 - alpha_bar).view(-1,1,1)

    #     diff = v_0 - (v_t/c0)
    #     var_b = (c1**2)/(c0**2)
    #     log_prob_b = -0.5*torch.sum(diff**2/var_b, dim=-1) - 0.5*3*torch.log(2*torch.tensor(np.pi)) - 3*torch.log((c1/c0).squeeze(-1))
    #     log_prob_b = torch.where(mask_generate, log_prob_b, torch.zeros_like(log_prob_b))
    #     return log_prob_b


class AminoacidCategoricalTransition(nn.Module):
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        N, L, K = c.size()
        c_ = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c_, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t, return_prob=False):
        """
        Forward transition p_F(x_t|x_0) for categorical data
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]
        c_noisy = (alpha_bar*c_0) + ((1-alpha_bar)/K)
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_0)
        x_t = self._sample(c_t)

        if return_prob:
            idx = x_t.unsqueeze(-1)
            p_x = torch.gather(c_t, dim=-1, index=idx).squeeze(-1)
            log_p_x = torch.log(p_x+1e-8)
            log_p_x = torch.where(mask_generate, log_p_x, torch.zeros_like(log_p_x))
            return c_t, x_t, log_p_x
        else:
            return c_t, x_t

    def add_noise_step_wise(self, x_t_1, mask_generate, t, return_prob=False):
        """
        Step-wise forward transition p_F(x_t|x_{t-1}) for categorical data
        
        q(x_j^t|x_j^{t-1}) = Multinomial((1 - β_t) · onehot(x_j^{t-1}) + β_t · \frac{1}{K} · 1)
        """
        N, L = x_t_1.size()
        K = self.num_classes
        
        # Convert input to one-hot representation
        c_t_1 = clampped_one_hot(x_t_1, num_classes=K).float()
        
        # Get beta value for current timestep
        beta_t = self.var_sched.betas[t][:, None, None]
        
        # Calculate transition probabilities
        # p(x_t|x_{t-1}) = (1-β_t)·onehot(x_{t-1}) + β_t/K
        c_noisy = (1 - beta_t) * c_t_1 + (beta_t/K)
        
        # Apply mask: keep original values for non-masked positions
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_t_1)
        
        # Sample from the categorical distribution
        x_t = self._sample(c_t)
        
        if return_prob:
            # Calculate log probabilities for masked positions
            idx = x_t.unsqueeze(-1)
            p_x = torch.gather(c_t, dim=-1, index=idx).squeeze(-1)
            log_p_x = torch.log(p_x + 1e-8)  # Add small epsilon for numerical stability
            log_p_x = torch.where(mask_generate, log_p_x, torch.zeros_like(log_p_x))
            #approx gaussian but not exact. log prob must be between -inf and 0
            log_prob = torch.clamp(log_p_x, min=-1e6, max=0)
            
            return c_t, x_t, log_p_x
        else:
            return c_t, x_t


    def posterior(self, x_t, x_0, t):
        """
        Posterior distribution p_B(x_0|x_t)
        """
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float()

        if x_0.dim() == 3:
            c_0 = x_0
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float()

        alpha = self.var_sched.alpha_bars[t][:, None, None]
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None]

        theta = ((alpha*c_t) + (1-alpha)/K) * ((alpha_bar*c_0) + (1-alpha_bar)/K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta


    def denoise(self, x_t, c_0_pred, mask_generate, t, return_prob=False):
        """
        Denoising step with option to return probability information
        
        Args:
            x_t: Current state tensor
            c_0_pred: Predicted initial state distribution
            mask_generate: Generation mask
            t: Current timestep
            return_prob: Whether to return probability information
        
        Returns:
            If return_prob=False:
                post: Posterior distribution
                x_next: Next state samples
            If return_prob=True:
                post: Posterior distribution
                x_next: Next state samples
                log_p_x: Log probabilities of selected samples
        """
        # Convert current state to one-hot representation
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()
        
        # Calculate posterior distribution
        post = self.posterior(c_t, c_0_pred, t=t)
        
        # Apply mask: keep original values for non-masked positions
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        
        # Sample next state
        x_next = self._sample(post)
        
        if return_prob:
            # Calculate log probabilities for the selected samples
            idx = x_next.unsqueeze(-1)
            p_x = torch.gather(post, dim=-1, index=idx).squeeze(-1)
            log_p_x = torch.log(p_x + 1e-8)  # Add small epsilon for numerical stability
            log_p_x = torch.where(mask_generate, log_p_x, torch.zeros_like(log_p_x))
            #approx gaussian but not exact. log prob must be between -inf and 0
            log_prob = torch.clamp(log_p_x, min=-1e6, max=0)
            
            return post, x_next, log_p_x
        else:
            return post, x_next
