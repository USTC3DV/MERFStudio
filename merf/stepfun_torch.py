import torch

def sorted_lookup(x, xp, fps):
  """Lookup `x` into locations `xp` , return indices and each `[fp]` value."""
  if not isinstance(fps, tuple):
    raise ValueError(f'Input `fps` must be a tuple, but is {type(fps)}.')

  
  idx = torch.vmap(lambda a, v: torch.searchsorted(a, v, side='right'))(
      xp.reshape([-1, xp.shape[-1]]), x.reshape([-1, x.shape[-1]])
  ).reshape(x.shape)
  idx1 = torch.minimum(idx, torch.ones_like(idx) * (xp.shape[-1] - 1))
  idx0 = torch.maximum(idx - 1, torch.zeros_like(idx))
  vals = []
  for fp in fps:
    fp0 = torch.take_along_dim(fp, idx0, dim=-1)
    fp1 = torch.take_along_dim(fp, idx1, dim=-1)
    vals.append((fp0, fp1))
  return (idx0, idx1), vals


def sorted_interp(
    x, xp, fp, eps=torch.finfo(torch.float32).eps ** 2
):
  """A version of interp() where xp and fp must be sorted."""
  (xp0, xp1), (fp0, fp1) = sorted_lookup(
      x, xp, (xp, fp)
  )[1]
  offset = torch.clip((x - xp0) / torch.maximum(torch.ones_like(xp1) * eps, xp1 - xp0), 0, 1)
  ret = fp0 + offset * (fp1 - fp0)
  return ret

def weight_to_pdf(t, w, eps=torch.finfo(torch.float32).eps ** 2):
  """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
  # print(w.shape)
  # print(torch.maximum(torch.ones_like(t[Ellipsis, 1:])*eps, (t[Ellipsis, 1:] - t[Ellipsis, :-1])).shape)
  return w / torch.maximum(torch.ones_like(t[Ellipsis, 1:])*eps, (t[Ellipsis, 1:] - t[Ellipsis, :-1]))


def pdf_to_weight(t, p):
  """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
  return p * (t[Ellipsis, 1:] - t[Ellipsis, :-1])


def max_dilate(t, w, dilation, domain=(-torch.inf, torch.inf)):
  """Dilate (via max-pooling) a non-negative step function."""
  t0 = t[Ellipsis, :-1] - dilation
  t1 = t[Ellipsis, 1:] + dilation
  t_dilate,_ = torch.sort(torch.concat([t, t0, t1], dim=-1), dim=-1)
  t_dilate = torch.clip(t_dilate, *domain)
  w_dilate,_ = torch.max(
      torch.where(
          (t0[Ellipsis, None, :] <= t_dilate[Ellipsis, None])
          & (t1[Ellipsis, None, :] > t_dilate[Ellipsis, None]),
          w[Ellipsis, None, :],
          0,
      ),
      dim=-1,
  )
  w_dilate =  w_dilate[Ellipsis, :-1]
  return t_dilate, w_dilate


def max_dilate_weights(
    t,
    w,
    dilation,
    domain=(-torch.inf, torch.inf),
    renormalize=False,
    eps=torch.finfo(torch.float32).eps,
):
  """Dilate (via max-pooling) a set of weights."""
  p = weight_to_pdf(t, w)
  t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
  w_dilate = pdf_to_weight(t_dilate, p_dilate)
  if renormalize:
    w_dilate /= torch.maximum(torch.ones_like(w_dilate)*eps, torch.sum(w_dilate, dim=-1, keepdims=True))
  return t_dilate, w_dilate


def integrate_weights(w):
  """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
  cumsum_w = torch.cumsum(w[Ellipsis, :-1], dim=-1)
  cw = torch.minimum(torch.ones_like(cumsum_w).to(cumsum_w.device),cumsum_w)
  shape = cw.shape[:-1] + (1,)
  # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
  cw0 = torch.concat([torch.zeros(shape).to(cw.device), cw, torch.ones(shape).to(cw.device)], dim=-1)
  return cw0


def invert_cdf(u, t, w_logits):
  """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
  # Compute the PDF and CDF for each weight vector.
  w = torch.softmax(w_logits, dim=-1)
  cw = integrate_weights(w)
  # Interpolate into the inverse CDF.
  t_new = sorted_interp(u, cw, t)
  return t_new


def sample(
    t,
    w_logits,
    num_samples,
    single_jitter=False,
    deterministic_center=False,
):
  """Piecewise-Constant PDF sampling from a step function.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    deterministic_center: bool, if False, when `rng` is None return samples that
      linspace the entire PDF. If True, skip the front and back of the linspace
      so that the centers of each PDF interval are returned.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  eps = torch.finfo(torch.float32).eps

  # Draw uniform samples.

  if deterministic_center:
    pad = 1 / (2 * num_samples)
    u = torch.linspace(pad, 1.0 - pad - eps, num_samples).to(t.device)
  else:
    u = torch.linspace(0, 1.0 - eps, num_samples).to(t.device)
  u = torch.broadcast_to(u, t.shape[:-1] + (num_samples,))


  return invert_cdf(u, t, w_logits)


def sample_intervals(
    t,
    w_logits,
    num_samples,
    single_jitter=False,
    domain=(-torch.inf, torch.inf),
):
  """Sample *intervals* (rather than points) from a step function.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of intervals to sample.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    domain: (minval, maxval), the range of valid values for `t`.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  if num_samples <= 1:
    raise ValueError(f'num_samples must be > 1, is {num_samples}.')

  # Sample a set of points from the step function.
  centers = sample(
    t, w_logits, num_samples, single_jitter, deterministic_center=True
  )

  # The intervals we return will span the midpoints of each adjacent sample.
  mid = (centers[Ellipsis, 1:] + centers[Ellipsis, :-1]) / 2

  # Each first/last fencepost is the reflection of the first/last midpoint
  # around the first/last sampled center. We clamp to the limits of the input
  # domain, provided by the caller.
  minval, maxval = domain
  first = torch.maximum(minval * torch.ones_like(centers[Ellipsis, :1]), 2 * centers[Ellipsis, :1] - mid[Ellipsis, :1])
  last = torch.minimum(maxval* torch.ones_like(centers[Ellipsis, :1]), 2 * centers[Ellipsis, -1:] - mid[Ellipsis, -1:])

  t_samples = torch.concat([first, mid, last], dim=-1)
  return t_samples
