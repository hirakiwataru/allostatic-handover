"""DreamerV3 Agent subclass with hidden-human-state auxiliary heads."""

from __future__ import annotations

from typing import Any

import numpy as np


def _imports() -> dict[str, Any]:
  import elements
  import embodied.jax
  import embodied.jax.nets as nn
  import jax
  import jax.numpy as jnp
  import ninjax as nj
  import optax
  from dreamerv3.agent import (
    Agent as DreamerAgent,
    concat,
    f32,
    i32,
    imag_loss,
    isimage,
    prefix,
    repl_loss,
    sample,
    sg,
  )
  from dreamerv3 import rssm

  return locals()


class AllostaticDreamerAgent(_imports()["DreamerAgent"]):
  """Exact DreamerV3 model with auxiliary heads for FSM/readiness/load.

  The inherited DreamerV3 JAX wrapper is still used. This subclass only changes
  the model part: labels are added as ``ext_space`` entries, not observations,
  so the encoder sees public observations only.
  """

  def __init__(self, obs_space, act_space, config):
    imp = _imports()
    elements = imp["elements"]
    embodied = __import__("embodied")
    nn = imp["nn"]
    rssm = imp["rssm"]
    optax = imp["optax"]

    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.num_human_states = int(getattr(config, "num_human_states", 6))

    exclude = ("is_first", "is_last", "is_terminal", "reward")
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
      "simple": rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name="enc")
    self.dyn = {
      "rssm": rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name="dyn")
    self.dec = {
      "simple": rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name="dec")

    self.feat2tensor = lambda x: imp["jnp"].concatenate(
      [
        nn.cast(x["deter"]),
        nn.cast(x["stoch"].reshape((*x["stoch"].shape[:-2], -1))),
      ],
      -1,
    )

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name="rew")
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name="con")

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(act_space, outs, **config.policy, name="pol")
    self.val = embodied.jax.MLPHead(scalar, **config.value, name="val")
    self.slowval = embodied.jax.SlowModel(
      embodied.jax.MLPHead(scalar, **config.value, name="slowval"),
      source=self.val,
      **config.slowvalue,
    )
    self.retnorm = embodied.jax.Normalize(**config.retnorm, name="retnorm")
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name="valnorm")
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name="advnorm")

    state_space = elements.Space(np.int32, (), 0, self.num_human_states)
    self.human_state_head = embodied.jax.MLPHead(
      state_space,
      "categorical",
      **config.aux_state_head,
      name="human_state",
    )
    self.readiness_head = embodied.jax.MLPHead(
      scalar,
      "mse",
      **config.aux_scalar_head,
      name="human_readiness",
    )
    self.load_head = embodied.jax.MLPHead(
      scalar,
      "mse",
      **config.aux_scalar_head,
      name="allostatic_load",
    )

    self.modules = [
      self.dyn,
      self.enc,
      self.dec,
      self.rew,
      self.con,
      self.pol,
      self.val,
      self.human_state_head,
      self.readiness_head,
      self.load_head,
    ]
    self.opt = embodied.jax.Optimizer(
      self.modules,
      self._make_opt(**config.opt),
      summary_depth=1,
      name="opt",
    )

    scales = self.config.loss_scales.copy()
    rec = scales.pop("rec")
    scales.update({k: rec for k in dec_space})
    scales.setdefault("human_state", 1.0)
    scales.setdefault("human_readiness", 1.0)
    scales.setdefault("allostatic_load", 0.2)
    self.scales = scales

  @property
  def ext_space(self):
    imp = _imports()
    elements = imp["elements"]
    spaces = super().ext_space
    spaces.update(
      {
        "human_state_id": elements.Space(np.int32, ()),
        "human_readiness": elements.Space(np.float32, ()),
        "allostatic_load_total": elements.Space(np.float32, ()),
      }
    )
    return spaces

  def train(self, carry, data):
    imp = _imports()
    jax = imp["jax"]
    obs, prevact, stepid, aux, carry = self._apply_allostatic_context(carry, data)
    metrics, (carry, entries, outs, mets) = self.opt(
      self.loss,
      carry,
      obs,
      prevact,
      aux,
      training=True,
      has_aux=True,
    )
    metrics.update(mets)
    self.slowval.update()
    outs = {}
    if self.config.replay_context:
      updates = imp["elements"].tree.flatdict(
        dict(stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2])
      )
      B, T = obs["is_first"].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
        (B, T),
        {k: v.shape for k, v in updates.items()},
      )
      outs["replay"] = updates
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def loss(self, carry, obs, prevact, aux, training):
    imp = _imports()
    jax = imp["jax"]
    jnp = imp["jnp"]
    f32 = imp["f32"]
    sg = imp["sg"]
    sample = imp["sample"]
    concat = imp["concat"]
    prefix = imp["prefix"]
    imag_loss = imp["imag_loss"]
    repl_loss = imp["repl_loss"]
    isimage = imp["isimage"]

    enc_carry, dyn_carry, dec_carry = carry
    reset = obs["is_first"]
    B, T = reset.shape
    losses = {}
    metrics = {}

    enc_carry, enc_entries, tokens = self.enc(enc_carry, obs, reset, training)
    dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
      dyn_carry,
      tokens,
      prevact,
      reset,
      training,
    )
    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.dec(dec_carry, repfeat, reset, training)
    feat_tensor = self.feat2tensor(repfeat)
    inp = sg(feat_tensor, skip=self.config.reward_grad)
    losses["rew"] = self.rew(inp, 2).loss(obs["reward"])
    con = f32(~obs["is_terminal"])
    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon
    losses["con"] = self.con(feat_tensor, 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))

    state_dist = self.human_state_head(feat_tensor, 2)
    readiness_dist = self.readiness_head(feat_tensor, 2)
    load_dist = self.load_head(feat_tensor, 2)
    state_target = aux["human_state_id"].astype(jnp.int32)
    readiness_target = aux["human_readiness"].astype(jnp.float32)
    load_target = aux["allostatic_load_total"].astype(jnp.float32)
    losses["human_state"] = state_dist.loss(state_target)
    losses["human_readiness"] = readiness_dist.loss(readiness_target)
    losses["allostatic_load"] = load_dist.loss(load_target)
    metrics["aux/human_state_acc"] = (state_dist.pred() == state_target).mean()
    metrics["aux/readiness_mae"] = jnp.abs(
      readiness_dist.pred() - readiness_target
    ).mean()
    metrics["aux/load_mae"] = jnp.abs(load_dist.pred() - load_target).mean()

    shapes = {k: v.shape for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length
    starts = self.dyn.starts(dyn_entries, dyn_carry, K)
    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
    first = jax.tree.map(
      lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])),
      repfeat,
    )
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    imag_inp = self.feat2tensor(imgfeat)
    los, imgloss_out, mets = imag_loss(
      imgact,
      self.rew(imag_inp, 2).pred(),
      self.con(imag_inp, 2).prob(1),
      self.pol(imag_inp, 2),
      self.val(imag_inp, 2),
      self.slowval(imag_inp, 2),
      self.retnorm,
      self.valnorm,
      self.advnorm,
      update=training,
      contdisc=self.config.contdisc,
      horizon=self.config.horizon,
      **self.config.imag_loss,
    )
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ("is_last", "is_terminal", "reward")]
      boot = imgloss_out["ret"][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
        lambda x: x[:, -K:],
        (feat, last, term, rew, boot),
      )
      repl_inp = self.feat2tensor(feat)
      los, _reploss_out, mets = repl_loss(
        last,
        term,
        rew,
        boot,
        self.val(repl_inp, 2),
        self.slowval(repl_inp, 2),
        self.valnorm,
        update=training,
        horizon=self.config.horizon,
        **self.config.repl_loss,
      )
      losses.update(los)
      metrics.update(prefix(mets, "reploss"))

    assert set(losses.keys()) == set(self.scales.keys()), (
      sorted(losses.keys()),
      sorted(self.scales.keys()),
    )
    metrics.update({f"loss/{k}": v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {"tokens": tokens, "repfeat": repfeat, "losses": losses}
    return loss, (carry, entries, outs, metrics)

  def _apply_allostatic_context(self, carry, data):
    imp = _imports()
    jax = imp["jax"]
    jnp = imp["jnp"]
    nn = imp["nn"]
    elements = imp["elements"]

    enc_carry, dyn_carry, dec_carry, prevact = carry
    carry3 = (enc_carry, dyn_carry, dec_carry)
    stepid = data["stepid"]
    obs = {k: data[k] for k in self.obs_space}
    aux = {
      "human_state_id": data["human_state_id"],
      "human_readiness": data["human_readiness"],
      "allostatic_load_total": data["allostatic_load_total"],
    }
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return obs, prevact, stepid, aux, carry3

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ("enc", "dyn", "dec")]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
      self.enc.truncate(lhs(entries[0]), enc_carry),
      self.dyn.truncate(lhs(entries[1]), dyn_carry),
      self.dec.truncate(lhs(entries[2]), dec_carry),
    )
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_aux = {k: rhs(v) for k, v in aux.items()}
    rep_prevact = {k: data[k][:, K - 1 : -1] for k in self.act_space}
    rep_stepid = rhs(stepid)
    first_chunk = data["consec"][:, 0] == 0
    carry3, obs, prevact, stepid = jax.tree.map(
      lambda normal, replay: nn.where(first_chunk, replay, normal),
      (carry3, rhs(obs), rhs(prevact), rhs(stepid)),
      (rep_carry, rep_obs, rep_prevact, rep_stepid),
    )
    aux = jax.tree.map(
      lambda normal, replay: nn.where(first_chunk, replay, normal),
      rhs(aux),
      rep_aux,
    )
    return obs, prevact, stepid, aux, carry3
