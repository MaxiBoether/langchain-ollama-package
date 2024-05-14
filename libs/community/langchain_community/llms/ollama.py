import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Union

import ollama
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Extra


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    generation_info = stream_response if stream_response.get("done") else None
    return GenerationChunk(
        text=stream_response["message"]["content"], generation_info=generation_info
    )


class OllamaEndpointNotFoundError(Exception):
    """Raised when the Ollama endpoint is not found."""


class _OllamaCommon(BaseLanguageModel):
    host: str = "http://localhost:11434"
    """Host the model is hosted under."""

    model: str = "llama2"
    """Model name to use."""

    timeout: Optional[int] = None
    """Timeout for the request stream"""

    @property
    def base_url(self) -> str:
        return self.host

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model": self.model}

    def _create_stream(
        self,
        api_url: str,
        payload: Any,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        options = kwargs.get("options", {})
        if "stop" in kwargs:
            options["stop"] = kwargs["stop"]

        if api_url.endswith("/generate"):
            yield from ollama.Client(host=self.host, timeout=self.timeout).generate(
                model=self.model,
                prompt=payload.get("prompt"),
                images=payload.get("images", []),
                stream=True,
                options=options,
            )
        elif api_url.endswith("/chat"):
            yield from ollama.Client(host=self.host, timeout=self.timeout).chat(
                model=self.model,
                messages=payload.get("messages", []),
                stream=True,
                options=options,
            )
        else:
            raise ValueError(f"Unsupported API URL: {api_url}")

    async def _acreate_stream(
        self,
        api_url: str,
        payload: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        options = kwargs.get("options", {})
        if "stop" in kwargs:
            options["stop"] = kwargs["stop"]

        if api_url.endswith("/generate"):
            async for chunk in await ollama.AsyncClient(
                host=self.host, timeout=self.timeout
            ).generate(
                model=self.model,
                prompt=payload.get("prompt"),
                images=payload.get("images", []),
                stream=True,
                options=options,
            ):
                yield chunk
        elif api_url.endswith("/chat"):
            async for chunk in await ollama.AsyncClient(
                host=self.host, timeout=self.timeout
            ).chat(
                model=self.model,
                messages=payload.get("messages", []),
                stream=True,
                options=options,
            ):
                yield chunk
        else:
            raise ValueError(f"Unsupported API URL: {api_url}")

    def _stream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        for stream_resp in self._create_stream(
            api_url=f"{self.host}/generate", payload={"prompt": prompt}, stop=stop, **kwargs
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _astream_with_aggregation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> GenerationChunk:
        final_chunk: Optional[GenerationChunk] = None
        async for stream_resp in self._acreate_stream(
            api_url=f"{self.host}/generate", payload={"prompt": prompt}, stop=stop, **kwargs
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk


class Ollama(BaseLLM, _OllamaCommon):
    """Ollama locally runs large language models."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "ollama-llm"

    def _generate(  # type: ignore[override]
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Ollama's generate endpoint."""
        generations = []
        for prompt in prompts:
            final_chunk = super()._stream_with_aggregation(
                prompt,
                stop=stop,
                images=images,
                run_manager=run_manager,
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    async def _agenerate(  # type: ignore[override]
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Call out to Ollama's generate endpoint asynchronously."""
        generations = []
        for prompt in prompts:
            final_chunk = await super()._astream_with_aggregation(
                prompt,
                stop=stop,
                images=images,
                run_manager=run_manager,  # type: ignore[arg-type]
                verbose=self.verbose,
                **kwargs,
            )
            generations.append([final_chunk])
        return LLMResult(generations=generations)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in self._create_stream(
            api_url=f"{self.host}/generate", payload={"prompt": prompt}, stop=stop, **kwargs
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        async for stream_resp in self._acreate_stream(
            api_url=f"{self.host}/generate", payload={"prompt": prompt}, stop=stop, **kwargs
        ):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk