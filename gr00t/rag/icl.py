"""Training-free in-context adaptation for a plain GR00T-N1.5 policy.

Memory: a handful of clean demonstrations per task are passed through the policy once.
For every stored frame we keep
  key      -- the frame's visual embedding: the VLM's hidden states at the image-token
              positions, mean-pooled and L2-normalised. The prompt is
              "<image-1><image-2>{instruction}" and the LLM is causal, so these states
              have not seen the instruction; the key is visual only.
  features -- the frame's full VLM token sequence (the backbone output the action head
              conditions on), stored raw, before vlln / vl_self_attention.
  actions  -- the chunk the policy's own action head generated for that frame
              (normalised space), not the demonstrator's ground truth.

Query: the new observation goes through the VLM, its visual key picks the top-k frames
by cosine similarity, and the retrieved context is concatenated onto the current input
of the frozen action head:
  - retrieved VLM tokens are appended to the DiT's cross-attention sequence;
  - retrieved action chunks are embedded with the head's own action encoder at the
    clean end of the flow (t = 1) and prepended to the DiT's token sequence
    [state, future tokens, noisy actions], where the interleaved self-attention blocks
    see them. The prediction is still read from the last action_horizon tokens.
No weights change; k = 0 reduces exactly to the stock policy.

Instruction augmentation (text_mode) is the alternative that leaves the action head's
input format untouched: retrieval picks a task from the memory, and text derived from that
context is appended to the OOD instruction before the normal forward pass.
  "retrieved" -- append the clean training instruction of the retrieved demonstrations
                 (majority vote over the top text_k frames).
  "generated" -- a generative VLM (scripts/rag/8_serve_hint_vlm.py) sees the retrieved
                 demonstration frame with its instruction and the current frame, and writes
                 a one-sentence restatement of the task, which is appended.
The text is decided on the first call for an instruction and reused for the rest of the
episode, so the prompt does not change mid-rollout.
"""
import base64
import io
import json
import time
import urllib.request
from collections import Counter

import numpy as np

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.policy import COMPUTE_DTYPE, Gr00tPolicy


def visual_key(backbone_features, input_ids, image_token_index):
    """(B, S, D) VLM states -> (B, D) unit vectors pooled over the image tokens."""
    selected = (input_ids == image_token_index).unsqueeze(-1).to(backbone_features.dtype)
    pooled = (backbone_features * selected).sum(1) / selected.sum(1).clamp_min(1)
    return F.normalize(pooled.float(), dim=-1)


class ICLMemory:
    def __init__(self, path, device):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.path = path
        self.keys = data["keys"].to(device)                     # (N, D) float32, unit norm
        self.actions = data["actions"].to(device)               # (N, H, A) generated
        self.features = [f.to(device) for f in data["features"]]  # N x (S_i, D) raw VLM
        self.tasks = data["tasks"]
        self.traj_ids = data["traj_ids"]
        self.frame_ids = data["frame_ids"]
        self.config = data["config"]

    def __len__(self):
        return len(self.tasks)

    def topk(self, query, k):
        scores = query @ self.keys.T                            # cosine: both unit norm
        return scores.topk(k, dim=-1)


def _context_tokens(head, ctx_actions, embodiment_id):
    """Retrieved (k, H, A) chunks -> (1, k*H, input_embedding_dim) clean action tokens."""
    k, H, _ = ctx_actions.shape
    clean_t = torch.full((k,), head.num_timestep_buckets - 1, device=ctx_actions.device)
    tokens = head.action_encoder(ctx_actions, clean_t, embodiment_id.expand(k))
    if head.config.add_pos_embed:
        pos_ids = torch.arange(H, dtype=torch.long, device=ctx_actions.device)
        tokens = tokens + head.position_embedding(pos_ids).unsqueeze(0)
    return tokens.reshape(1, k * H, -1)


@torch.no_grad()
def icl_get_action(head, backbone_output, action_input, ctx_features, ctx_actions,
                   use_features=True, use_actions=True):
    """FlowmatchingActionHead.get_action with retrieved context concatenated (batch 1)."""
    backbone_output = head.process_backbone_output(backbone_output)
    vl_embs = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    if vl_embs.shape[0] != 1:
        raise ValueError("in-context inference is implemented for batch size 1")

    if use_features:
        # Each retrieved frame goes through vlln / vl_self_attention on its own, exactly as
        # it would have as the current input, then joins the cross-attention sequence.
        processed = []
        for feats in ctx_features:
            ctx = BatchFeature(data={
                "backbone_features": feats.unsqueeze(0).to(vl_embs.dtype),
                "backbone_attention_mask": torch.ones(
                    1, feats.shape[0], dtype=torch.long, device=vl_embs.device),
            })
            processed.append(head.process_backbone_output(ctx).backbone_features)
        vl_embs = torch.cat([vl_embs, *processed], dim=1)

    prefix = (_context_tokens(head, ctx_actions.to(vl_embs.dtype), embodiment_id)
              if use_actions else None)

    state_features = head.state_encoder(action_input.state, embodiment_id)
    future_tokens = head.future_tokens.weight.unsqueeze(0)
    H, A = head.config.action_horizon, head.config.action_dim
    actions = torch.randn((1, H, A), dtype=vl_embs.dtype, device=vl_embs.device)

    num_steps = head.num_inference_timesteps
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t_discretized = int(step / float(num_steps) * head.num_timestep_buckets)
        timesteps = torch.full((1,), t_discretized, device=vl_embs.device)
        action_features = head.action_encoder(actions, timesteps, embodiment_id)
        if head.config.add_pos_embed:
            pos_ids = torch.arange(H, dtype=torch.long, device=vl_embs.device)
            action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)

        parts = [state_features, future_tokens, action_features]
        if prefix is not None:
            parts.insert(0, prefix)
        sa_embs = torch.cat(parts, dim=1)

        model_output = head.model(
            hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timesteps)
        pred = head.action_decoder(model_output, embodiment_id)
        actions = actions + dt * pred[:, -H:]
    return BatchFeature(data={"action_pred": actions})


def _png_b64(image):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class ICLPolicy(Gr00tPolicy):
    """Gr00tPolicy whose action head is conditioned on retrieved in-context examples."""

    LANGUAGE_KEY = "annotation.human.action.task_description"

    def __init__(self, *args, memory_path=None, k=0, use_features=True, use_actions=True,
                 retrieval_log=None, text_mode="none", text_k=5, hint_url=None,
                 demo_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.use_features = use_features
        self.use_actions = use_actions
        self.text_mode = text_mode
        self.text_k = text_k
        self.hint_url = hint_url
        self.demo_dataset = demo_dataset
        self.text_cache = {}
        needs_memory = k > 0 or text_mode != "none"
        self.memory = ICLMemory(memory_path, self.device) if needs_memory else None
        self.retrieval_log = retrieval_log
        self.image_token_index = self.model.backbone.eagle_model.image_token_index

    def get_action(self, observations):
        if self.text_mode != "none":
            observations = self._augment_instruction(observations)
        return super().get_action(observations)

    def _augment_instruction(self, observations):
        instruction = str(np.asarray(observations[self.LANGUAGE_KEY]).reshape(-1)[0])
        if instruction not in self.text_cache:
            self.text_cache[instruction] = self._context_text(observations, instruction)
        observations = dict(observations)
        observations[self.LANGUAGE_KEY] = [self.text_cache[instruction]]
        return observations

    def _context_text(self, observations, instruction):
        from gr00t.model.policy import unsqueeze_dict_values

        normalized = self.apply_transforms(unsqueeze_dict_values(dict(observations)))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            key = self.encode(normalized)[3]
        scores, indices = self.memory.topk(key, self.text_k)
        picked, scores = indices[0].tolist(), scores[0].tolist()
        task = Counter(self.memory.tasks[i] for i in picked).most_common(1)[0][0]
        hint = task
        if self.text_mode == "generated":
            best = next(i for i in picked if self.memory.tasks[i] == task)
            hint = self._generate_hint(observations, instruction, task, best)
        text = f"{instruction}. {hint}"
        if self.retrieval_log:
            with open(self.retrieval_log, "a") as fh:
                fh.write(json.dumps({
                    "time": time.time(), "text_mode": self.text_mode,
                    "instruction": instruction, "voted_task": task, "hint": hint,
                    "prompt": text,
                    "retrieved_task": [self.memory.tasks[i] for i in picked],
                    "cosine": scores,
                }) + "\n")
        print(f"[icl-text] {instruction!r} -> {text!r}", flush=True)
        return text

    def _generate_hint(self, observations, instruction, task, index):
        step = self.demo_dataset.get_step_data(
            self.memory.traj_ids[index], self.memory.frame_ids[index])
        request = {
            "reference_image": _png_b64(step["video.image"][0]),
            "current_image": _png_b64(np.asarray(observations["video.image"])[0]),
            "reference_instruction": task, "instruction": instruction,
        }
        req = urllib.request.Request(
            self.hint_url, data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read())
        if "hint" not in payload:
            raise RuntimeError(f"hint service failed: {payload}")
        return payload["hint"]

    def encode(self, normalized_input):
        """Backbone pass shared by memory building and inference."""
        backbone_inputs, action_inputs = self.model.prepare_input(normalized_input)
        backbone_outputs = self.model.backbone(backbone_inputs)
        key = visual_key(backbone_outputs.backbone_features,
                         backbone_inputs["eagle_input_ids"], self.image_token_index)
        return backbone_inputs, backbone_outputs, action_inputs, key

    def _get_action_from_normalized_input(self, normalized_input):
        head = self.model.action_head
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            _, backbone_outputs, action_inputs, key = self.encode(normalized_input)
            if self.k == 0:
                model_pred = head.get_action(backbone_outputs, action_inputs)
            else:
                scores, indices = self.memory.topk(key, self.k)
                picked = indices[0].tolist()
                model_pred = icl_get_action(
                    head, backbone_outputs, action_inputs,
                    [self.memory.features[i] for i in picked],
                    self.memory.actions[picked],
                    self.use_features, self.use_actions)
                self._log(normalized_input, picked, scores[0].tolist())
        return model_pred["action_pred"].float()

    def _log(self, normalized_input, picked, scores):
        if not self.retrieval_log:
            return
        with open(self.retrieval_log, "a") as fh:
            fh.write(json.dumps({
                "time": time.time(),
                "retrieved_task": [self.memory.tasks[i] for i in picked],
                "retrieved_traj": [self.memory.traj_ids[i] for i in picked],
                "retrieved_frame": [self.memory.frame_ids[i] for i in picked],
                "cosine": scores,
            }) + "\n")
