from __future__ import annotations

import json
from typing import Protocol
from urllib import error, request

from .config import AppConfig, ModelProfile


class LLMClient(Protocol):
    def complete_with_fallback(self, prompt: str, max_tokens: int = 700) -> str:
        ...


class LlamaCppClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def complete(
        self,
        prompt: str,
        model: ModelProfile | None = None,
        max_tokens: int = 700,
    ) -> str:
        selected = model or self._config.primary_model
        payload = {
            "prompt": prompt,
            "temperature": selected.temperature,
            "n_predict": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._config.llama_cpp_endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"llama.cpp request failed: {exc}") from exc

        parsed = json.loads(body)
        text = parsed.get("content") or parsed.get("response") or ""
        return text.strip()

    def complete_with_fallback(self, prompt: str, max_tokens: int = 700) -> str:
        try:
            return self.complete(prompt, model=self._config.primary_model, max_tokens=max_tokens)
        except Exception as primary_error:
            try:
                return self.complete(prompt, model=self._config.fallback_model, max_tokens=max_tokens)
            except Exception as fallback_error:
                if self._config.mock_on_llm_error:
                    return (
                        "MOCK_RESPONSE: llama.cpp endpoint unavailable. "
                        "This is a fallback stub for local workflow verification.\n"
                        f"Primary error: {primary_error}\n"
                        f"Fallback error: {fallback_error}\n"
                        f"Prompt excerpt: {prompt[:200]}"
                    )
                raise


class OllamaClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def complete(
        self,
        prompt: str,
        model: ModelProfile | None = None,
        max_tokens: int = 700,
    ) -> str:
        selected = model or self._config.primary_model
        payload = {
            "model": selected.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": selected.temperature,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._config.ollama_endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(f"ollama request failed: {exc}") from exc

        parsed = json.loads(body)
        text = parsed.get("response") or parsed.get("content") or ""
        return text.strip()

    def complete_with_fallback(self, prompt: str, max_tokens: int = 700) -> str:
        try:
            return self.complete(prompt, model=self._config.primary_model, max_tokens=max_tokens)
        except Exception as primary_error:
            try:
                return self.complete(prompt, model=self._config.fallback_model, max_tokens=max_tokens)
            except Exception as fallback_error:
                if self._config.mock_on_llm_error:
                    return (
                        "MOCK_RESPONSE: ollama endpoint unavailable. "
                        "This is a fallback stub for local workflow verification.\n"
                        f"Primary error: {primary_error}\n"
                        f"Fallback error: {fallback_error}\n"
                        f"Prompt excerpt: {prompt[:200]}"
                    )
                raise


def build_llm_client(config: AppConfig) -> LLMClient:
    provider = (config.llm_provider or "").strip().lower()
    if provider == "llama_cpp":
        return LlamaCppClient(config)
    return OllamaClient(config)
