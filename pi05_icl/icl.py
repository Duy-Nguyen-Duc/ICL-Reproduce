"""The gr00t/rag/icl.py method, ported to pi0.5 (openpi, JAX). Runs in the openpi venv.

Memory (built by scripts/pi05/1_build_memory.py from the same demo frames as GR00T's):
  key      -- visual embedding: the SigLIP image tokens of the valid views, mean-pooled and
              L2-normalised. pi0.5's prefix attends bidirectionally, so the LLM's states
              at image positions have already seen the instruction; the SigLIP tokens are
              the visual-only embedding.
  tokens   -- the frame's prefix embedding sequence (images + tokenised prompt), i.e. the
              VLM input; the always-masked right-wrist slot is dropped.
  actions  -- the chunk pi0.5 itself generates for that frame (normalised model space).

Concatenation (k > 0), the counterpart of appending to GR00T's cross-attention memory and
DiT token sequence:
  - each retrieved prefix is run through the VLM on its own and its per-layer KV cache is
    appended to the current observation's, so the action expert attends to both;
  - retrieved action chunks are embedded with the expert's own action_in_proj and placed
    before the noisy action tokens in the same bidirectional suffix block. The prediction
    is read from the last action_horizon tokens.
Instruction augmentation (text_mode) matches the GR00T version: vote the retrieved task
over the top text_k frames and append it ("retrieved"), or append a sentence written by
the hint VLM from the retrieved demo frame and the current view ("generated").
k = 0 with text_mode "none" calls the stock openpi Policy.infer unchanged.
"""
import base64
import dataclasses
import io
import json
import time
import urllib.request
from collections import Counter

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.policies import policy_config
from openpi.training import config as _config

IMAGE_TOKENS = 256            # SigLIP So400m/14 at 224px
DROPPED_SLOT = 2              # right_wrist_0_rgb: always masked for LIBERO (pi0.5)


def encode(model, observation):
    """Prefix embeddings, their mask, and the visual key for a batch of observations."""
    observation = _model.preprocess_observation(None, observation, train=False)
    tokens, mask, _ = model.embed_prefix(observation)
    n_img = IMAGE_TOKENS * len(observation.images)
    weights = mask[:, :n_img, None].astype(jnp.float32)
    pooled = (tokens[:, :n_img].astype(jnp.float32) * weights).sum(1) / weights.sum(1)
    key = pooled / jnp.linalg.norm(pooled, axis=-1, keepdims=True)
    return tokens, mask, key


def _time_cond(model, time, batch):
    emb = posemb_sincos(jnp.broadcast_to(time, batch), model.action_in_proj.out_features,
                        min_period=4e-3, max_period=4.0)
    emb = nnx.swish(model.time_mlp_in(emb))
    return nnx.swish(model.time_mlp_out(emb))


def sample_icl(model, rng, observation, ctx_tokens, ctx_mask, ctx_actions, *,
               num_steps=10, use_features=True, use_actions=True):
    """Pi0.sample_actions with retrieved context concatenated (batch 1, k contexts)."""
    observation = _model.preprocess_observation(None, observation, train=False)
    H, D = model.action_horizon, model.action_dim
    noise = jax.random.normal(rng, (1, H, D))
    llm = model.PaliGemma.llm

    tokens, mask, ar = model.embed_prefix(observation)
    _, kv_cache = llm([tokens, None], mask=make_attn_mask(mask, ar),
                      positions=jnp.cumsum(mask, axis=1) - 1)
    kv_mask = mask
    if use_features:
        for i in range(ctx_tokens.shape[0]):
            c_tok, c_mask = ctx_tokens[i][None], ctx_mask[i][None]
            _, c_kv = llm([c_tok, None], mask=make_attn_mask(c_mask, jnp.zeros(c_mask.shape[1], bool)),
                          positions=jnp.cumsum(c_mask, axis=1) - 1)
            kv_cache = tuple(jnp.concatenate([a, b], axis=2) for a, b in zip(kv_cache, c_kv))
            kv_mask = jnp.concatenate([kv_mask, c_mask], axis=1)

    prefix_len = jnp.sum(mask, axis=-1)[:, None]
    ctx_act = (model.action_in_proj(ctx_actions.reshape(1, -1, D).astype(noise.dtype))
               if use_actions else None)
    dt = -1.0 / num_steps

    def step(carry):
        x_t, t = carry
        act = model.action_in_proj(x_t)
        suffix = act if ctx_act is None else jnp.concatenate([ctx_act, act], axis=1)
        s_len = suffix.shape[1]
        s_mask = jnp.ones((1, s_len), dtype=bool)
        s_ar = jnp.array([True] + [False] * (s_len - 1))
        full_mask = jnp.concatenate([
            einops.repeat(kv_mask, "b p -> b s p", s=s_len), make_attn_mask(s_mask, s_ar)], axis=-1)
        positions = prefix_len + jnp.cumsum(s_mask, axis=-1) - 1
        (_, out), _ = llm([None, suffix], mask=full_mask, positions=positions,
                          kv_cache=kv_cache, adarms_cond=[None, _time_cond(model, t, 1)])
        v_t = model.action_out_proj(out[:, -H:])
        return x_t + dt * v_t, t + dt

    x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
    return x_0


def drop_masked_slot(tokens, mask):
    """Remove the always-masked right-wrist image tokens before storing a prefix."""
    lo, hi = DROPPED_SLOT * IMAGE_TOKENS, (DROPPED_SLOT + 1) * IMAGE_TOKENS
    keep = np.r_[0:lo, hi:tokens.shape[1]]
    return tokens[:, keep], mask[:, keep]


def _png_b64(image):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@dataclasses.dataclass
class ICLMemory:
    keys: jax.Array          # (N, 2048) float32
    tokens: jax.Array        # (N, P, 2048) bfloat16
    mask: jax.Array          # (N, P) bool
    actions: jax.Array       # (N, H, 32) float32, normalised
    tasks: list
    traj_ids: list
    frame_ids: list

    @classmethod
    def load(cls, path, with_tokens=True):
        z = np.load(path, allow_pickle=False)
        tokens = jnp.asarray(z["tokens"].view(jnp.bfloat16)) if with_tokens else None
        return cls(jnp.asarray(z["keys"]), tokens, jnp.asarray(z["mask"]) if with_tokens else None,
                   jnp.asarray(z["actions"]), z["task"].tolist(), z["traj_id"].tolist(),
                   z["frame_id"].tolist())


class Pi05ICL:
    def __init__(self, checkpoint_dir, *, memory_path=None, k=0, use_features=True,
                 use_actions=True, text_mode="none", text_k=5, hint_url=None,
                 demo_frames=None, retrieval_log=None, seed=0):
        self.config = _config.get_config("pi05_libero")
        self.policy = policy_config.create_trained_policy(self.config, checkpoint_dir)
        self.model = self.policy._model
        self.k, self.use_features, self.use_actions = k, use_features, use_actions
        self.text_mode, self.text_k, self.hint_url = text_mode, text_k, hint_url
        self.demo_frames = demo_frames
        self.retrieval_log = retrieval_log
        self.text_cache = {}
        self.rng = jax.random.key(seed)
        needs_memory = k > 0 or text_mode != "none"
        self.memory = ICLMemory.load(memory_path, with_tokens=k > 0) if needs_memory else None
        self._encode = nnx.jit(encode)
        self._sample_icl = nnx.jit(sample_icl, static_argnames=(
            "num_steps", "use_features", "use_actions"))

    # -- openpi plumbing -------------------------------------------------------------
    def prepare(self, obs):
        inputs = self.policy._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def generate(self, obs):
        """Stock action generation, normalised model space (for building the memory)."""
        _, observation = self.prepare(obs)
        self.rng, rng = jax.random.split(self.rng)
        return self.policy._sample_actions(rng, observation, **self.policy._sample_kwargs)

    def unnormalize(self, inputs, actions):
        out = self.policy._output_transform({"state": inputs["state"][0],
                                             "actions": np.asarray(actions[0])})
        return out["actions"]

    # -- inference -------------------------------------------------------------------
    def infer(self, obs):
        if self.text_mode != "none":
            obs = dict(obs)
            prompt = obs["prompt"]
            if prompt not in self.text_cache:
                self.text_cache[prompt] = self._context_text(obs)
            obs["prompt"] = self.text_cache[prompt]
        if self.k == 0:
            return self.policy.infer(obs)["actions"]

        inputs, observation = self.prepare(obs)
        _, _, key = self._encode(self.model, observation)
        scores, idx = jax.lax.top_k(key @ self.memory.keys.T, self.k)
        picked = np.asarray(idx[0]).tolist()
        self.rng, rng = jax.random.split(self.rng)
        actions = self._sample_icl(
            self.model, rng, observation, self.memory.tokens[idx[0]], self.memory.mask[idx[0]],
            self.memory.actions[idx[0]], use_features=self.use_features,
            use_actions=self.use_actions)
        self._log({"retrieved_task": [self.memory.tasks[i] for i in picked],
                   "retrieved_traj": [self.memory.traj_ids[i] for i in picked],
                   "retrieved_frame": [self.memory.frame_ids[i] for i in picked],
                   "cosine": np.asarray(scores[0]).tolist()})
        return self.unnormalize(inputs, actions)

    def _context_text(self, obs):
        _, observation = self.prepare(obs)
        _, _, key = self._encode(self.model, observation)
        scores, idx = jax.lax.top_k(key @ self.memory.keys.T, self.text_k)
        picked = np.asarray(idx[0]).tolist()
        task = Counter(self.memory.tasks[i] for i in picked).most_common(1)[0][0]
        hint = task
        if self.text_mode == "generated":
            best = next(i for i in picked if self.memory.tasks[i] == task)
            hint = self._generate_hint(obs, task, best)
        text = f"{obs['prompt']}. {hint}"
        self._log({"text_mode": self.text_mode, "instruction": obs["prompt"],
                   "voted_task": task, "hint": hint, "prompt": text,
                   "retrieved_task": [self.memory.tasks[i] for i in picked],
                   "cosine": np.asarray(scores[0]).tolist()})
        print(f"[icl-text] {obs['prompt']!r} -> {text!r}", flush=True)
        return text

    def _generate_hint(self, obs, task, index):
        request = {"reference_image": _png_b64(self.demo_frames["image"][index]),
                   "current_image": _png_b64(obs["observation/image"]),
                   "reference_instruction": task, "instruction": obs["prompt"]}
        req = urllib.request.Request(self.hint_url, data=json.dumps(request).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        if "hint" not in payload:
            raise RuntimeError(f"hint service failed: {payload}")
        return payload["hint"]

    def _log(self, record):
        if self.retrieval_log:
            with open(self.retrieval_log, "a") as fh:
                fh.write(json.dumps({"time": time.time(), **record}) + "\n")
