from __future__ import annotations

from typing import Optional, Type, List, Union, Any, Dict
from types import SimpleNamespace

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LLMClient:
    """Centralized LLM client using LangChain with reusable methods (fixed)."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "o4-mini",
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _make_llm(self, temperature: float, max_tokens: Optional[int] = None) -> ChatOpenAI:
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "model": self.model,          # OpenAI model name OR Azure deployment name
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return ChatOpenAI(**kwargs)

    @staticmethod
    def _safe_fallback(schema_class: Type[BaseModel], error: Exception) -> BaseModel:
        """
        Best-effort fallback for PEN-MATCH style schemas.
        Uses model_construct() to avoid crashing if schema has required fields.
        """
        payload = {
            "decision": "NO_MATCH",
            "confidence": 0.0,
            "reasons": [f"LLM analysis failed: {str(error)}"],
            "mismatches": {},
        }
        try:
            return schema_class(**payload)
        except Exception:
            # Pydantic v2: bypass validation (keeps pipeline alive)
            return schema_class.model_construct(**payload)

    def with_structured_output(self, schema_class: Type[BaseModel]):
        """Structured output from a single user prompt (no system message)."""

        def invoke(user_prompt: str, temperature: float = 0.1, max_tokens: Optional[int] = None) -> BaseModel:
            try:
                llm = self._make_llm(temperature=temperature, max_tokens=max_tokens)

                prompt = ChatPromptTemplate.from_messages([
                    ("user", "{user_prompt}")
                ])

                chain = prompt | llm.with_structured_output(schema_class)

                result = chain.invoke({"user_prompt": user_prompt})

                if isinstance(result, BaseModel):
                    return result
                if isinstance(result, dict):
                    return schema_class(**result)
                # Unexpected type: convert to string and fail-safe
                raise TypeError(f"Unexpected structured output type: {type(result)}")

            except Exception as e:
                print(f"Error in structured LLM call: {e}")
                return self._safe_fallback(schema_class, e)

        # IMPORTANT: return invoke as an instance attribute (no method binding)
        return SimpleNamespace(invoke=invoke)

    def with_structured_output_and_system(self, schema_class: Type[BaseModel]):
        """Structured output with a system prompt + user prompt."""

        def invoke(
            system_prompt: str,
            user_prompt: str,
            temperature: float = 0.1,
            max_tokens: Optional[int] = None,
        ) -> BaseModel:
            try:
                llm = self._make_llm(temperature=temperature, max_tokens=max_tokens)

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "{system_prompt}"),
                    ("user", "{user_prompt}"),
                ])

                chain = prompt | llm.with_structured_output(schema_class)

                result = chain.invoke({
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                })

                if isinstance(result, BaseModel):
                    return result
                if isinstance(result, dict):
                    return schema_class(**result)
                raise TypeError(f"Unexpected structured output type: {type(result)}")

            except Exception as e:
                print(f"Error in structured LLM call with system prompt: {e}")
                return self._safe_fallback(schema_class, e)

        return SimpleNamespace(invoke=invoke)

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Simple text generation (string output)."""
        try:
            llm = self._make_llm(temperature=temperature, max_tokens=max_tokens)
            chain = ChatPromptTemplate.from_messages([("user", "{prompt}")]) | llm | StrOutputParser()
            return chain.invoke({"prompt": prompt})
        except Exception as e:
            print(f"Error in text generation: {e}")
            return f"Error: {str(e)}"

    def analyze_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Analysis with system context (string output)."""
        try:
            llm = self._make_llm(temperature=temperature, max_tokens=max_tokens)
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", "{system_prompt}"),
                    ("user", "{user_prompt}"),
                ])
                | llm
                | StrOutputParser()
            )
            return chain.invoke({"system_prompt": system_prompt, "user_prompt": user_prompt})
        except Exception as e:
            print(f"Error in contextual analysis: {e}")
            return f"Error: {str(e)}"

    def create_custom_chain(
        self,
        prompt_template: Union[str, ChatPromptTemplate],
        output_parser=None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ):
        """Create a reusable LangChain chain."""
        try:
            llm = self._make_llm(temperature=temperature, max_tokens=max_tokens)
            prompt = prompt_template if isinstance(prompt_template, ChatPromptTemplate) else ChatPromptTemplate.from_template(prompt_template)
            if output_parser:
                return prompt | llm | output_parser
            return prompt | llm | StrOutputParser()
        except Exception as e:
            print(f"Error creating custom chain: {e}")
            return None

    def batch_analyze(self, prompts: List[str], temperature: float = 0.1, max_tokens: Optional[int] = None) -> List[str]:
        """Batch process multiple prompts (string outputs)."""
        try:
            llm = self._make_llm(temperature=temperature, max_tokens=max_tokens)
            chain = ChatPromptTemplate.from_messages([("user", "{prompt}")]) | llm | StrOutputParser()
            inputs = [{"prompt": p} for p in prompts]
            return chain.batch(inputs)
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return [f"Error: {str(e)}" for _ in prompts]
