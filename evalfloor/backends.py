"""Where a typed decision actually gets made.

Two implementations ship. `LocalBackend` reads option logits from any causal
LM in one prefill -- the trick every open System One reproduction uses -- so
the whole search loop runs offline at zero marginal cost. `JevBackend` calls
the hosted model, and exists for the final validation pass only.

Keeping the loop on a local backend is not just thrift. TypeSafe's customer
agreement forbids using Service output to train a model to imitate it or to
build a competing service, so a search loop that optimises text against a
local open model, and merely CHECKS the winner against the hosted one, stays
clearly on the right side of that line.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Protocol


class Backend(Protocol):
    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        """name -> {"probabilities": {...}, "choice": str} or {"noul": float}."""
        ...


def render_state(state: dict) -> str:
    lines = []
    for k, v in state.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


class LocalBackend:
    """Score options by next-token logits from one prefill.

    For a `choice`, options are presented as lettered lines and the logits of
    the letter tokens are softmaxed against each other -- so the model never
    generates, and the answer is a probability vector over exactly the
    allowed options. For a `noul`, the same is done over yes/no.

    This is deliberately the simplest thing that works. It is not trying to
    match a purpose-trained decision model's accuracy; it is trying to give
    the SEARCH a cheap, deterministic, offline signal. Whether a schema found
    this way transfers to the hosted model is an empirical question the
    library answers rather than assumes -- see `validate_transfer`.
    """

    @staticmethod
    def _labels(n: int) -> list[str]:
        """Option markers. Letters read better; numbers scale past 26.

        The hosted model allows up to 255 options, so a 26-letter scheme
        would silently truncate a real task (banking77 has 77 intents) and
        report the resulting damage as a schema problem. Switching wholesale
        rather than mixing keeps every option's marker one token.
        """
        if n <= 26:
            return [chr(65 + i) for i in range(n)]
        return [str(i + 1) for i in range(n)]

    def __init__(self, model="Qwen/Qwen2.5-0.5B-Instruct", device=None, max_len=3072):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float32 if self.device == "cpu" else torch.float16
        ).to(self.device).eval()
        self.max_len = max_len
        self._cache: dict[str, Any] = {}

    def _score_tokens(self, prompt: str, surfaces: list[str]) -> list[float]:
        """Probability over exactly the allowed options, from one prefill.

        Only the resulting probabilities are cached, never the logits row.
        Caching the full vocabulary tensor -- and on the accelerator, at that
        -- costs ~600 KB per distinct prompt and exhausted a 18 GB device
        part-way through the first real run. The answer for a given
        (prompt, options) pair is all anything downstream ever needs.
        """
        key = (prompt, tuple(surfaces))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ids = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=self.max_len).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**ids).logits[0, -1]
        vals = []
        for s in surfaces:
            tid = self.tok.encode(s, add_special_tokens=False)
            vals.append(float(logits[tid[0]]) if tid else -1e9)
        del logits
        out = self.torch.softmax(self.torch.tensor(vals), dim=0).tolist()
        if len(self._cache) < 200000:
            self._cache[key] = out
        return out

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        head = render_state(state)
        out: dict[str, Any] = {}
        for name, q in questions.items():
            if q.get("type") == "choice":
                opts = list((q.get("criteria") or {}).items())
                if not opts:
                    out[name] = {"probabilities": {}, "choice": None}
                    continue
                marks = self._labels(len(opts))
                body = "\n".join(f"{marks[i]}) {k}: {v}"
                                 for i, (k, v) in enumerate(opts))
                p = self._score_tokens(
                    f"{head}\n\n{q.get('instructions','')}\n{body}\n\nAnswer:",
                    [f" {m}" for m in marks])
                probs = {k: p[i] for i, (k, _v) in enumerate(opts)}
                out[name] = {"probabilities": probs,
                             "choice": max(probs, key=probs.get)}
            else:
                p = self._score_tokens(
                    f"{head}\n\n{q.get('instructions','')}\n\nAnswer (yes or no):",
                    [" yes", " no"])
                out[name] = {"noul": p[0]}
        return out


class JevBackend:
    """The hosted model. Used for validation, not for the search loop."""

    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model="typesafe/jev-1.13", api_key=None, timeout=90):
        self.model = model
        self.timeout = timeout
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise SystemExit("set OPENROUTER_API_KEY to use JevBackend")
        self.calls = 0

    def decide(self, state: dict, questions: dict) -> dict[str, Any]:
        body = {"model": self.model, "state": state, "questions": questions}
        req = urllib.request.Request(
            self.ENDPOINT, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.key,
                     "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self.calls += 1
            return json.loads(r.read().decode()).get("answers", {})
