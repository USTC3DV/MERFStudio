
import numpy as np
import torch


def pos_enc(x, min_deg, max_deg, append_identity=True):
  """The positional encoding used by the original NeRF paper."""
  scales = 2 ** torch.arange(min_deg, max_deg).to(x.device)
  shape = x.shape[:-1] + (-1,)
  scaled_x = torch.reshape((x[Ellipsis, None, :] * scales[:, None]), shape)
  four_feat = torch.sin(
      torch.cat([scaled_x, scaled_x + 0.5 * torch.pi], dim=-1)
  )
  if append_identity:
    return torch.cat([x] + [four_feat], dim=-1)
  else:
    return four_feat


def piecewise_warp_fwd(x, eps=torch.finfo(torch.float32).eps):
  """A piecewise combo of linear and reciprocal to allow t_near=0."""
  return torch.where(x < 1, 0.5 * x, 1 - 0.5 / torch.maximum(torch.ones_like(x)*eps, x))


def piecewise_warp_inv(x, eps=torch.finfo(torch.float32).eps):
  """The inverse of `piecewise_warp_fwd`."""
  return torch.where(x < 0.5, 2 * x, 0.5 / torch.maximum(torch.ones_like(x)*eps, 1 - x))


def s_to_t(s, t_near, t_far):
  """Convert normalized distances ([0,1]) to world distances ([t_near, t_far])."""
  s_near, s_far = [piecewise_warp_fwd(x) for x in (t_near, t_far)]
  return piecewise_warp_inv(s * s_far + (1 - s) * s_near)


def contract(x):
  """The contraction function we proposed in MERF."""
  # For more info check out MERF: Memory-Efficient Radiance Fields for Real-time
  # View Synthesis in Unbounded Scenes: https://arxiv.org/abs/2302.12249,
  # Section 4.2
  # After contraction points lie within [-2,2]^3.
  x_abs = torch.abs(x)
  # Clamping to 1 produces correct scale inside |x| < 1.
  x_max = torch.maximum(1.0 * torch.ones_like(x_abs), torch.amax(x_abs, dim=-1, keepdims=True))
  scale = 1 / x_max  # no divide by 0 because of previous maximum(1, ...)
  z = scale * x
  # The above produces coordinates like (x/z, y/z, 1)
  # but we still need to replace the "1" with \pm (2-1/z).
  idx = torch.argmax(x_abs, dim=-1, keepdims=True)
  negative = torch.take_along_dim(z, idx, dim=-1) < 0
  o = torch.where(negative, -2 + scale, 2 - scale)
  # Select the final values by coordinate.
  ival_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
  ival = torch.arange(x.shape[-1]).reshape(ival_shape).to(idx.device)
  result = torch.where(x_max <= 1, x, torch.where(ival == idx, o, z))
  # result = x
  return result


def stepsize_in_squash(x, d, v):
  """Computes step size in contracted space."""
  # Approximately computes s such that ||c(x+d*s) - c(x)||_2 = v, where c is
  # the contraction function, i.e. we often need to know by how much (s) the ray
  # needs to be advanced to get an advancement of v in contracted space.
  #
  # The further we are from the scene's center the larger steps in world space
  # we have to take to get the same advancement in contracted space.
  contract_0_grad = torch.func.grad(lambda x: contract(x)[0])
  contract_1_grad = torch.func.grad(lambda x: contract(x)[1])
  contract_2_grad = torch.func.grad(lambda x: contract(x)[2])
  # print(x.shape)
  # print(d.shape)
  def helper(x, d):
    return torch.sqrt(
        d.dot(contract_0_grad(x)) ** 2
        + d.dot(contract_1_grad(x)) ** 2
        + d.dot(contract_2_grad(x)) ** 2
    )

  return v / torch.vmap(helper)(x, d)
