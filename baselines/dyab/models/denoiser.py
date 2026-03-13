import math
import torch
from torch import nn
import torch.nn.functional as F

from .modules.pair_transition import PairTransition
from .modules.triangular_attention import (
	TriangleAttentionStartingNode,
	TriangleAttentionEndingNode,
)
from .modules.triangular_multiplicative_update import (
	TriangleMultiplicationOutgoing,
	TriangleMultiplicationIncoming,
)
from .modules.dropout import DropoutRowwise, DropoutColumnwise

from .modules.invariant_point_attention import InvariantPointAttention
from .modules.structure_transition import StructureTransition
from .modules.backbone_update import BackboneUpdate


def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

def dihedral(p, eps=1e-10):
    # p: [*, 4, 3]

    # [*, 3]
    u1 = p[..., 1, :] - p[..., 0, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u3 = p[..., 3, :] - p[..., 2, :]

    # [*, 3]
    u1xu2 = torch.cross(u1, u2, dim=-1)
    u2xu3 = torch.cross(u2, u3, dim=-1)

    # [*]
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5
    u1xu2_norm = (eps + torch.sum(u1xu2 ** 2, dim=-1)) ** 0.5
    u2xu3_norm = (eps + torch.sum(u2xu3 ** 2, dim=-1)) ** 0.5

    # [*]
    cos_enc = torch.einsum('...d,...d->...', u1xu2, u2xu3)/ (u1xu2_norm * u2xu3_norm)
    sin_enc = torch.einsum('...d,...d->...', u2, torch.cross(u1xu2, u2xu3, dim=-1)) /  (u2_norm * u1xu2_norm * u2xu3_norm)

    return torch.stack([cos_enc, sin_enc], dim=-1)


def compute_frenet_frames(x, mask, eps=1e-10):
    # x: [b, n_res, 3]

    t = x[:, 1:] - x[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    b = torch.cross(t[:, :-1], t[:, 1:])
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    n = torch.cross(b, t[:, 1:])

    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    # TODO: recheck correctness of this implementation
    rots = []
    for i in range(mask.shape[0]):
        rots_ = torch.eye(3).unsqueeze(0).repeat(mask.shape[1], 1, 1)
        length = torch.sum(mask[i]).int()
        rots_[1:length-1] = tbn[i, :length-2]
        rots_[0] = rots_[1]
        rots_[length-1] = rots_[length-2]
        rots.append(rots_)
    rots = torch.stack(rots, dim=0).to(x.device)

    return rots


def get_template_fn(template):
	if template == 'v1':
		return v1, 1

def v1(t):

	# [b, n_res, n_res, 1]
	d = distance(torch.stack([
		t.trans.unsqueeze(2).repeat(1, 1, t.trans.shape[1], 1), # Ca_1
		t.trans.unsqueeze(1).repeat(1, t.trans.shape[1], 1, 1), # Ca_2
	], dim=-2)).unsqueeze(-1)

	return d


def sinusoidal_encoding(v, N, D):
	# v: [*]

	# [D]
	k = torch.arange(1, D+1).to(v.device)

	# [*, D]
	sin_div_term = N ** (2 * k / D)
	sin_div_term = sin_div_term.view(*((1, ) * len(v.shape) + (len(sin_div_term), )))
	sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

	# [*, D]
	cos_div_term = N ** (2 * (k - 1) / D)
	cos_div_term = cos_div_term.view(*((1, ) * len(v.shape) + (len(cos_div_term), )))
	cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

	# [*, D]
	enc = torch.zeros_like(sin_enc).to(v.device)
	enc[..., 0::2] = cos_enc[..., 0::2]
	enc[..., 1::2] = sin_enc[..., 1::2]

	return enc


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class SingleFeatureNet(nn.Module):

	def __init__(self,
		c_s,
		# n_timestep,
		c_pos_emb,
		c_timestep_emb
	):
		super(SingleFeatureNet, self).__init__()

		self.c_s = c_s
		# self.n_timestep = n_timestep
		self.c_pos_emb = c_pos_emb
		self.c_timestep_emb = c_timestep_emb

		self.linear = nn.Linear(self.c_pos_emb + self.c_timestep_emb, self.c_s)

	def forward(self, ts, timesteps, mask):
		# s: [b]

		b, max_n_res, device = ts.shape[0], ts.shape[1], timesteps.device

		# [b, n_res, c_pos_emb]
		pos_emb = sinusoidal_encoding(torch.arange(max_n_res).to(device), max_n_res, self.c_pos_emb)
		pos_emb = pos_emb.unsqueeze(0).repeat([b, 1, 1])
		pos_emb = pos_emb * mask.unsqueeze(-1)

		# [b, n_res, c_timestep_emb]
		timestep_emb = get_time_embedding(
            timesteps,
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, max_n_res, 1)
		timestep_emb = timestep_emb * mask.unsqueeze(-1)
		# timestep_emb = sinusoidal_encoding(timesteps.view(b, 1), self.n_timestep, self.c_timestep_emb)
		# timestep_emb = timestep_emb.repeat(1, max_n_res, 1)
		# timestep_emb = timestep_emb * mask.unsqueeze(-1)

		return self.linear(torch.cat([
			pos_emb,
			timestep_emb
		], dim=-1))


class PairFeatureNet(nn.Module):

	def __init__(self, c_s, c_p, relpos_k, template_type):
		super(PairFeatureNet, self).__init__()

		self.c_s = c_s
		self.c_p = c_p

		self.linear_s_p_i = nn.Linear(c_s, c_p)
		self.linear_s_p_j = nn.Linear(c_s, c_p)

		self.relpos_k = relpos_k
		self.n_bin = 2 * relpos_k + 1
		self.linear_relpos = nn.Linear(self.n_bin, c_p)

		self.template_fn, c_template = get_template_fn(template_type)
		self.linear_template = nn.Linear(c_template, c_p)

	def relpos(self, r):
		# AlphaFold 2 Algorithm 4 & 5
		# Based on OpenFold utils/tensor_utils.py
		# Input: [b, n_res]

		# [b, n_res, n_res]
		d = r[:, :, None] - r[:, None, :]

		# [n_bin]
		v = torch.arange(-self.relpos_k, self.relpos_k + 1).to(r.device)
		
		# [1, 1, 1, n_bin]
		v_reshaped = v.view(*((1,) * len(d.shape) + (len(v),)))

		# [b, n_res, n_res]
		b = torch.argmin(torch.abs(d[:, :, :, None] - v_reshaped), dim=-1)

		# [b, n_res, n_res, n_bin]
		oh = nn.functional.one_hot(b, num_classes=len(v)).float()

		# [b, n_res, n_res, c_p]
		p = self.linear_relpos(oh)

		return p

	def template(self, t):
		return self.linear_template(self.template_fn(t))

	def forward(self, s, t, p_mask):
		# Input: [b, n_res, c_s]

		# [b, n_res, c_p]
		p_i = self.linear_s_p_i(s)
		p_j = self.linear_s_p_j(s)

		# [b, n_res, n_res, c_p]
		p = p_i[:, :, None, :] + p_j[:, None, :, :]

		# [b, n_res]
		r = torch.arange(s.shape[1]).unsqueeze(0).repeat(s.shape[0], 1).to(s.device)

		# [b, n_res, n_res, c_p]
		p += self.relpos(r)
		p += self.template(t)

		# [b, n_res, n_res, c_p]
		p *= p_mask.unsqueeze(-1)

		return p
	

class PairTransformLayer(nn.Module):

	def __init__(self,
		c_p,
		include_mul_update,
		include_tri_att,
		c_hidden_mul,
		c_hidden_tri_att,
		n_head_tri,
		tri_dropout,
		pair_transition_n,
	):
		super(PairTransformLayer, self).__init__()

		self.tri_mul_out = TriangleMultiplicationOutgoing(
			c_p,
			c_hidden_mul
		) if include_mul_update else None

		self.tri_mul_in = TriangleMultiplicationIncoming(
			c_p,
			c_hidden_mul
		) if include_mul_update else None

		self.tri_att_start = TriangleAttentionStartingNode(
			c_p,
			c_hidden_tri_att,
			n_head_tri
		) if include_tri_att else None

		self.tri_att_end = TriangleAttentionEndingNode(
			c_p,
			c_hidden_tri_att,
			n_head_tri
		) if include_tri_att else None

		self.pair_transition = PairTransition(
			c_p,
			pair_transition_n
		)

		self.dropout_row_layer = DropoutRowwise(tri_dropout)
		self.dropout_col_layer = DropoutColumnwise(tri_dropout)

	def forward(self, inputs):
		p, p_mask = inputs
		if self.tri_mul_out is not None:
			p = p + self.dropout_row_layer(self.tri_mul_out(p, p_mask))
			p = p + self.dropout_row_layer(self.tri_mul_in(p, p_mask))
		if self.tri_att_start is not None:
			p = p + self.dropout_row_layer(self.tri_att_start(p, p_mask))
			p = p + self.dropout_col_layer(self.tri_att_end(p, p_mask))
		p = p + self.pair_transition(p, p_mask)
		p = p * p_mask.unsqueeze(-1)
		outputs = (p, p_mask)
		return outputs

class PairTransformNet(nn.Module):

	def __init__(self,
		c_p,
		n_pair_transform_layer,
		include_mul_update,
		include_tri_att,
		c_hidden_mul,
		c_hidden_tri_att,
		n_head_tri,
		tri_dropout,
		pair_transition_n
	):
		super(PairTransformNet, self).__init__()

		layers = [
			PairTransformLayer(
				c_p,
				include_mul_update,
				include_tri_att,
				c_hidden_mul,
				c_hidden_tri_att,
				n_head_tri,
				tri_dropout,
				pair_transition_n
			)
			for _ in range(n_pair_transform_layer)
		]

		self.net = nn.Sequential(*layers)

	def forward(self, p, p_mask):
		p, _ = self.net((p, p_mask))
		return p


class StructureLayer(nn.Module):

	def __init__(self,
		c_s,
		c_p,
		c_hidden_ipa,
		n_head,
		n_qk_point,
		n_v_point,
		ipa_dropout,
		n_structure_transition_layer,
		structure_transition_dropout
	):
		super(StructureLayer, self).__init__()

		self.ipa = InvariantPointAttention(
			c_s,
			c_p,
			c_hidden_ipa,
			n_head,
			n_qk_point,
			n_v_point
		)
		self.ipa_dropout = nn.Dropout(ipa_dropout)
		self.ipa_layer_norm = nn.LayerNorm(c_s)

		# Built-in dropout and layer norm
		self.transition = StructureTransition(
			c_s,
			n_structure_transition_layer, 
			structure_transition_dropout
		)
		
		# backbone update
		self.bb_update = BackboneUpdate(c_s)

	def forward(self, inputs):
		s, p, t, mask = inputs
		s = s + self.ipa(s, p, t, mask)
		s = self.ipa_dropout(s)
		s = self.ipa_layer_norm(s)
		s = self.transition(s)
		t = t.compose(self.bb_update(s))
		outputs = (s, p, t, mask)
		return outputs


class StructureNet(nn.Module):

	def __init__(self,
		c_s,
		c_p,
		n_structure_layer,
		n_structure_block,
		c_hidden_ipa,
		n_head_ipa,
		n_qk_point,
		n_v_point,
		ipa_dropout,
		n_structure_transition_layer,
		structure_transition_dropout		
	):
		super(StructureNet, self).__init__()

		self.n_structure_block = n_structure_block

		layers = [
			StructureLayer(
				c_s, c_p,
				c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout, 
				n_structure_transition_layer, structure_transition_dropout
			)
			for _ in range(n_structure_layer)
		]
		self.net = nn.Sequential(*layers)

	def forward(self, s, p, t, mask):
		for block_idx in range(self.n_structure_block):
			s, p, t, mask = self.net((s, p, t, mask))
		return t


class Denoiser(nn.Module):

	def __init__(self,
		c_s=128, c_p=128, # n_timestep,
		c_pos_emb=128, c_timestep_emb=128,
		relpos_k=32, template_type='v1',
		n_pair_transform_layer=5, include_mul_update=True, include_tri_att=False,
		c_hidden_mul=128, c_hidden_tri_att=32, n_head_tri=4, tri_dropout=0.25, pair_transition_n=4,
		n_structure_layer=5, n_structure_block=1,
		c_hidden_ipa=16, n_head_ipa=12, n_qk_point=4, n_v_point=8, ipa_dropout=0.1,
		n_structure_transition_layer=1, structure_transition_dropout=0.1
	):
		super(Denoiser, self).__init__()

		self.single_feature_net = SingleFeatureNet(
			c_s,
			# n_timestep,
			c_pos_emb,
			c_timestep_emb
		)
		
		self.pair_feature_net = PairFeatureNet(
			c_s,
			c_p,
			relpos_k,
			template_type
		)

		self.pair_transform_net = PairTransformNet(
			c_p,
			n_pair_transform_layer,
			include_mul_update,
			include_tri_att,
			c_hidden_mul,
			c_hidden_tri_att,
			n_head_tri,
			tri_dropout,
			pair_transition_n
		) if n_pair_transform_layer > 0 else None

		self.structure_net = StructureNet(
			c_s,
			c_p,
			n_structure_layer,
			n_structure_block,
			c_hidden_ipa,
			n_head_ipa,
			n_qk_point,
			n_v_point,
			ipa_dropout,
			n_structure_transition_layer,
			structure_transition_dropout
		)

	def forward(self, ts, timesteps, mask):
		p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
		s = self.single_feature_net(ts, timesteps, mask)
		p = self.pair_feature_net(s, ts, p_mask)
		if self.pair_transform_net is not None:
			p = self.pair_transform_net(p, p_mask)
		ts = self.structure_net(s, p, ts, mask)
		return ts