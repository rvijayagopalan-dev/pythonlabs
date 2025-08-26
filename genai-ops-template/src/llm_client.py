from __future__ import annotations
import os
from typing import List, Dict, Any

import tiktoken
from openai import OpenAI, AzureOpenAI

from . import config

class LLMClient:
    def __init__(self):
        if config.LLM_PROVIDER == "azure":
            self.client = AzureOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version="2024-05-01-preview",
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            )
            self.chat_model = config.AZURE_OPENAI_CHAT_DEPLOYMENT
            self.embed_model = config.AZURE_OPENAI_EMBED_DEPLOYMENT
        else:
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            self.chat_model = config.OPENAI_MODEL
            self.embed_model = config.OPENAI_EMBED_MODEL

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
        )
        choice = resp.choices[0]
        out = {
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "usage": resp.usage.model_dump() if hasattr(resp, "usage") else {},
            "model": self.chat_model,
        }
        return out

    def embed(self, texts: List[str]) -> List[List[float]]:
        if config.LLM_PROVIDER == "azure":
            resp = self.client.embeddings.create(
                model=self.embed_model, input=texts
            )
        else:
            resp = self.client.embeddings.create(
                model=self.embed_model, input=texts
            )
        return [d.embedding for d in resp.data]
