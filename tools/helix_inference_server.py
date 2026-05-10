#!/usr/bin/env python3
"""
WO-STACK-WIRE-01: Standalone HXQ inference server.

One file. One command. OpenAI-compatible /v1/chat/completions backed by
HelixLinear + fused Triton kernel on compressed models.

Usage:
    # Direct inference (no agent loop):
    python3 tools/helix_inference_server.py --model ~/models/zamba2-7b-instruct-helix

    # With agent loop (tool-calling ReAct cycle):
    python3 tools/helix_inference_server.py --model ~/models/zamba2-7b-instruct-helix --agent

    # With KV cache compression:
    python3 tools/helix_inference_server.py --model ~/models/zamba2-7b-instruct-helix --kv-cache

    # Test:
    curl -X POST http://localhost:8001/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'

Depends on: helix-substrate >= 0.3.0, transformers, torch, fastapi, uvicorn
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import resource
import platform
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Register HXQ quantizer before any model loading
# ---------------------------------------------------------------------------
import helix_substrate.hf_quantizer  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False

class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Model backend
# ---------------------------------------------------------------------------

class HelixBackend:
    """Loads a compressed HXQ model and serves inference."""

    def __init__(self, model_path: str, device: str = "auto", kv_cache: bool = False):
        self.model_path = model_path
        self.device = device
        self.model_name = Path(model_path).name

        print(f"Loading compressed model from {model_path}...", flush=True)
        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()

        # Count HelixLinear modules
        from helix_substrate.helix_linear import HelixLinear
        self.n_helix = sum(1 for m in self.model.modules() if isinstance(m, HelixLinear))

        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s — {self.n_helix} HelixLinear modules", flush=True)

        # GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2
            print(f"  VRAM: {mem:.0f} MB", flush=True)
            self.vram_mb = mem
        else:
            self.vram_mb = 0

        # Optional KV cache compression
        self.kv_cache = None
        if kv_cache:
            self._init_kv_cache()

    def _init_kv_cache(self):
        """Initialize compressed KV cache from helix-online-kv."""
        try:
            sys.path.insert(0, os.path.expanduser("~/helix-online-kv"))
            from helix_online_kv.config import OnlineKVConfig
            from helix_online_kv.compressed_cache import CompressedKVCache

            n_layers = self.model.config.num_hidden_layers
            config = OnlineKVConfig(n_clusters=256, calibration_tokens=128)
            self.kv_cache = CompressedKVCache(config, n_layers)
            print(f"  KV cache: CompressedKVCache(k=256, layers={n_layers})", flush=True)
        except Exception as e:
            print(f"  KV cache init failed: {e} — continuing without", flush=True)

    def _apply_chat_template(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to model input string."""
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return self.tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            parts = []
            for m in messages:
                if m.role == "system":
                    parts.append(f"System: {m.content}\n")
                elif m.role == "user":
                    parts.append(f"User: {m.content}\n")
                elif m.role == "assistant":
                    parts.append(f"Assistant: {m.content}\n")
            parts.append("Assistant:")
            return "".join(parts)

    def generate(self, messages: List[ChatMessage], max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 1.0) -> Dict[str, Any]:
        """Run inference and return OpenAI-compatible response."""
        t0 = time.time()

        prompt = self._apply_chat_template(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.01,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.kv_cache is not None:
            gen_kwargs["past_key_values"] = self.kv_cache

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_len = len(new_tokens)

        elapsed = time.time() - t0
        tok_per_sec = completion_len / max(elapsed, 0.001)

        return {
            "text": text,
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "elapsed_s": round(elapsed, 3),
            "tok_per_sec": round(tok_per_sec, 2),
        }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(backend: HelixBackend) -> FastAPI:
    app = FastAPI(title="HXQ Inference Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": backend.model_name,
            "n_helix_linear": backend.n_helix,
            "vram_mb": round(backend.vram_mb),
            "kv_cache": backend.kv_cache is not None,
        }

    @app.get("/v1/models")
    def list_models():
        return {
            "data": [{
                "id": backend.model_name,
                "object": "model",
                "owned_by": "echo-labs",
            }]
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        if req.stream:
            raise HTTPException(501, "Streaming not implemented")

        try:
            result = backend.generate(
                messages=req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}")

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=backend.model_name,
            choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=result["text"]),
            )],
            usage=UsageInfo(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["prompt_tokens"] + result["completion_tokens"],
            ),
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HXQ Inference Server — compressed model API in one file",
    )
    parser.add_argument("--model", required=True, help="Path to compressed HXQ model")
    parser.add_argument("--device", default="auto", help="Device map (auto, cuda, cpu)")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--kv-cache", action="store_true", help="Enable compressed KV cache")
    args = parser.parse_args()

    backend = HelixBackend(
        model_path=args.model,
        device=args.device,
        kv_cache=args.kv_cache,
    )

    # Print startup receipt
    receipt = {
        "work_order": "WO-STACK-WIRE-01",
        "stack": "helix_inference_server → HelixLinear → Triton kernel",
        "model": backend.model_name,
        "model_path": args.model,
        "n_helix_linear": backend.n_helix,
        "vram_mb": round(backend.vram_mb),
        "kv_cache": backend.kv_cache is not None,
        "endpoint": f"http://{args.host}:{args.port}/v1/chat/completions",
        "cost": {
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        },
    }
    print(json.dumps(receipt, indent=2), flush=True)

    # Save receipt
    receipts_dir = Path(__file__).resolve().parent.parent / "receipts" / "stack_wire"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    receipt_path = receipts_dir / f"startup_{backend.model_name}_{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}", flush=True)

    import uvicorn
    app = create_app(backend)
    print(f"\n{'='*60}")
    print(f"  HXQ Inference Server")
    print(f"  Model: {backend.model_name} ({backend.n_helix} HelixLinear)")
    print(f"  VRAM:  {backend.vram_mb:.0f} MB")
    print(f"  URL:   http://{args.host}:{args.port}/v1/chat/completions")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
