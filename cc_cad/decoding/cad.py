"""Context-Aware Decoding (CAD) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def fuse_logits(
    logits_context: torch.Tensor, logits_prior: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Fuse logits per CAD rule."""
    return (1.0 + alpha) * logits_context - alpha * logits_prior


def apply_temperature_top_p(
    logits: torch.Tensor, temperature: float = 1.0, top_p: Optional[float] = None
) -> torch.Tensor:
    """Apply temperature and top-p truncation, returning normalized probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    probs.zero_().scatter_(dim=-1, index=sorted_idx, src=sorted_probs)
    return probs


@dataclass
class CADGeneration:
    text: str
    tokens: List[int]
    logprobs: Optional[List[float]] = None
    approximation: Optional[str] = None


class APIClient:
    """API client interface for black-box LLMs."""

    def generate_with_context(self, prompt: str, evidence: str) -> CADGeneration:
        raise NotImplementedError

    def generate_without_context(self, prompt: str) -> CADGeneration:
        raise NotImplementedError


class StubAPIClient(APIClient):
    """Stub implementation that returns empty answers."""

    def generate_with_context(self, prompt: str, evidence: str) -> CADGeneration:
        return CADGeneration(text="", tokens=[], logprobs=None)

    def generate_without_context(self, prompt: str) -> CADGeneration:
        return CADGeneration(text="", tokens=[], logprobs=None)


class CADOpenGenerator:
    """CAD decoding for open-weight models."""

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.device = device
        self.max_positions = getattr(self.model.config, "n_positions", None)

    def generate(
        self,
        prompt: str,
        evidence: str,
        alpha: float,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        mode: str = "greedy",
        num_beams: int = 4,
    ) -> CADGeneration:
        """Generate text via CAD fusion of context and prior logits."""
        context_prompt = f"{prompt}\n\nEvidence:\n{evidence}\n\nAnswer:"
        prior_prompt = f"{prompt}\n\nAnswer:"
        if mode == "greedy":
            return self._generate_greedy(
                context_prompt, prior_prompt, alpha, max_new_tokens, temperature, top_p
            )
        if mode == "beam":
            return self._generate_beam(
                context_prompt,
                prior_prompt,
                alpha,
                max_new_tokens,
                temperature,
                top_p,
                num_beams,
            )
        raise ValueError(f"Unknown mode: {mode}")

    def _generate_greedy(
        self,
        context_prompt: str,
        prior_prompt: str,
        alpha: float,
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
    ) -> CADGeneration:
        context_ids = self._encode_prompt(context_prompt)
        prior_ids = self._encode_prompt(prior_prompt)
        generated: List[int] = []
        logprobs: List[float] = []
        for _ in range(max_new_tokens):
            context_logits = self.model(context_ids).logits[:, -1, :]
            prior_logits = self.model(prior_ids).logits[:, -1, :]
            fused = fuse_logits(context_logits, prior_logits, alpha)
            probs = apply_temperature_top_p(fused, temperature, top_p)
            next_id = int(torch.argmax(probs, dim=-1).item())
            logprobs.append(float(torch.log(probs[0, next_id] + 1e-12).item()))
            if next_id == self.tokenizer.eos_token_id:
                break
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=self.device)
            context_ids = self._append_and_trim(context_ids, next_tensor)
            prior_ids = self._append_and_trim(prior_ids, next_tensor)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return CADGeneration(text=text, tokens=generated, logprobs=logprobs)

    def _generate_beam(
        self,
        context_prompt: str,
        prior_prompt: str,
        alpha: float,
        max_new_tokens: int,
        temperature: float,
        top_p: Optional[float],
        num_beams: int,
    ) -> CADGeneration:
        context_ids = self._encode_prompt(context_prompt)
        prior_ids = self._encode_prompt(prior_prompt)
        beams = [([], 0.0, context_ids, prior_ids)]
        for _ in range(max_new_tokens):
            new_beams: List[Tuple[List[int], float, torch.Tensor, torch.Tensor]] = []
            for tokens, score, ctx_ids, pr_ids in beams:
                ctx_logits = self.model(ctx_ids).logits[:, -1, :]
                pr_logits = self.model(pr_ids).logits[:, -1, :]
                fused = fuse_logits(ctx_logits, pr_logits, alpha)
                probs = apply_temperature_top_p(fused, temperature, top_p)
                top_scores, top_ids = torch.topk(probs, k=num_beams, dim=-1)
                for prob, token_id in zip(top_scores[0], top_ids[0]):
                    token_int = int(token_id.item())
                    new_score = score + float(torch.log(prob + 1e-12).item())
                    if token_int == self.tokenizer.eos_token_id:
                        new_beams.append((tokens, new_score, ctx_ids, pr_ids))
                        continue
                    next_tensor = torch.tensor([[token_int]], device=self.device)
                    new_ctx = self._append_and_trim(ctx_ids, next_tensor)
                    new_pr = self._append_and_trim(pr_ids, next_tensor)
                    new_beams.append((tokens + [token_int], new_score, new_ctx, new_pr))
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:num_beams]
            beams = new_beams
        best_tokens, best_score, _, _ = beams[0]
        text = self.tokenizer.decode(best_tokens, skip_special_tokens=True)
        return CADGeneration(text=text, tokens=best_tokens, logprobs=None)

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if self.max_positions is None:
            return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        max_len = self.max_positions - 1
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_len
        ).input_ids.to(self.device)
        return input_ids

    def _append_and_trim(self, input_ids: torch.Tensor, next_token: torch.Tensor) -> torch.Tensor:
        updated = torch.cat([input_ids, next_token], dim=-1)
        if self.max_positions is None:
            return updated
        max_len = self.max_positions - 1
        if updated.shape[1] <= max_len:
            return updated
        return updated[:, -max_len:]


def cad_api_generate(
    client: APIClient,
    prompt: str,
    evidence: str,
    alpha: float,
) -> CADGeneration:
    """Approximate CAD via API calls with/without evidence."""
    with_ctx = client.generate_with_context(prompt, evidence)
    without_ctx = client.generate_without_context(prompt)
    if with_ctx.logprobs and without_ctx.logprobs and with_ctx.tokens == without_ctx.tokens:
        fused_logprobs = []
        for lp_ctx, lp_prior in zip(with_ctx.logprobs, without_ctx.logprobs):
            fused_logprobs.append((1.0 + alpha) * lp_ctx - alpha * lp_prior)
        return CADGeneration(
            text=with_ctx.text,
            tokens=with_ctx.tokens,
            logprobs=fused_logprobs,
            approximation="token_fusion",
        )
    # Fallback: sentence-level reranking by total logprob if available
    if with_ctx.logprobs and without_ctx.logprobs:
        score_ctx = float(np.sum(with_ctx.logprobs))
        score_prior = float(np.sum(without_ctx.logprobs))
        chosen = with_ctx if score_ctx >= score_prior else without_ctx
        return CADGeneration(
            text=chosen.text,
            tokens=chosen.tokens,
            logprobs=chosen.logprobs,
            approximation="rerank",
        )
    return CADGeneration(text=with_ctx.text, tokens=with_ctx.tokens, logprobs=with_ctx.logprobs, approximation="unknown")
