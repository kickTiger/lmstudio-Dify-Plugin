import json
import logging
import time
from decimal import Decimal
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import requests
from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    FetchFrom,
    I18nObject,
    ModelPropertyKey,
    ModelType,
    PriceConfig,
    PriceType,
)
from dify_plugin.entities.model.text_embedding import (
    EmbeddingUsage,
    TextEmbeddingResult,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

logger = logging.getLogger(__name__)


class LmstudioEmbeddingModel(TextEmbeddingModel):
    """
    Model class for LM Studio text embedding model.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :param input_type: input type
        :return: embeddings result
        """
        # Set up base URL
        base_url = credentials.get("base_url", "")
        if base_url and not base_url.endswith("/"):
            base_url += "/"
        
        # Configure OpenAI-compatible client for LM Studio
        from openai import OpenAI
        client = OpenAI(base_url=f"{base_url}v1", api_key="lm-studio")
        
        # Calculate tokens before potentially truncating
        embedding_used_tokens = self.get_num_tokens(model, credentials, texts)
        used_tokens = sum(embedding_used_tokens)
        
        # Process embeddings
        context_size = self._get_context_size(model, credentials)
        processed_texts = []
        
        # Truncate texts if they exceed context size
        for text, num_tokens in zip(texts, embedding_used_tokens):
            if num_tokens >= context_size:
                cutoff = int(np.floor(len(text) * (context_size / num_tokens)))
                processed_texts.append(text[0:cutoff])
            else:
                processed_texts.append(text)
        
        # Get embeddings through OpenAI compatible endpoint
        try:
            response = client.embeddings.create(
                model=model,
                input=processed_texts,
                encoding_format="float"
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            # Calculate usage
            usage = self._calc_response_usage(
                model=model, credentials=credentials, tokens=used_tokens
            )
            
            return TextEmbeddingResult(embeddings=embeddings, usage=usage, model=model)
            
        except Exception as e:
            # Map error to appropriate type
            raise self._invoke_error_mapping(e)

    def get_num_tokens(
        self, model: str, credentials: dict, texts: list[str]
    ) -> list[int]:
        """
        Approximate number of tokens for given texts using GPT2 tokenizer

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :return: list of token counts for each text
        """
        return [self._get_num_tokens_by_gpt2(text) for text in texts]

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return: None if valid, raises exception otherwise
        """
        try:
            # Verify base URL
            base_url = credentials.get("base_url", "")
            if not base_url:
                raise CredentialsValidateFailedError("Base URL is required")
                
            if not base_url.endswith("/"):
                base_url += "/"
                
            # Use requests to check connection
            response = requests.get(
                urljoin(base_url, "v1/models"),
                timeout=5
            )
            
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Failed to connect to LM Studio server: {response.status_code}"
                )
                
            # Try a simple embedding request to validate
            self._invoke(model=model, credentials=credentials, texts=["ping"])
            
        except InvokeError as ex:
            raise CredentialsValidateFailedError(
                f"An error occurred during credentials validation: {ex.description}"
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(
                f"An error occurred during credentials validation: {str(ex)}"
            )

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        Generate custom model entities from credentials
        
        :param model: model name
        :param credentials: model credentials
        :return: model entity
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.TEXT_EMBEDDING,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(
                    credentials.get("context_size", 4096)
                ),
                ModelPropertyKey.MAX_CHUNKS: 1,
            },
            parameter_rules=[],
            pricing=PriceConfig(
                input=Decimal(credentials.get("input_price", 0)),
                unit=Decimal(credentials.get("unit", 0)),
                currency=credentials.get("currency", "USD"),
            ),
        )
        return entity

    def _calc_response_usage(
        self, model: str, credentials: dict, tokens: int
    ) -> EmbeddingUsage:
        """
        Calculate response usage

        :param model: model name
        :param credentials: model credentials
        :param tokens: input tokens
        :return: usage
        """
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens,
        )
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at,
        )
        return usage

    def _invoke_error_mapping(self, error: Exception) -> Exception:
        """
        Map error from invocation to standardized error.

        :param error: exception from invocation
        :return: standardized exception with context
        """
        error_messages = str(error)
        
        if isinstance(error, requests.exceptions.ConnectTimeout):
            return InvokeConnectionError("Connection timeout error: " + error_messages)
        elif isinstance(error, requests.exceptions.ReadTimeout):
            return InvokeConnectionError("Read timeout error: " + error_messages)
        elif isinstance(error, requests.exceptions.ConnectionError):
            return InvokeConnectionError("Connection error: " + error_messages)
        elif "Unauthorized" in error_messages or "API key" in error_messages:
            return InvokeAuthorizationError("Authorization error: " + error_messages)
        elif "Bad request" in error_messages or "Invalid request" in error_messages:
            return InvokeBadRequestError("Bad request error: " + error_messages)
        elif "Too many requests" in error_messages or "Rate limit" in error_messages:
            return InvokeRateLimitError("Rate limit error: " + error_messages)
        elif "Server error" in error_messages or "Internal server error" in error_messages:
            return InvokeServerUnavailableError("Server error: " + error_messages)
        
        # Default fallback for other errors
        return InvokeError("Error invoking LM Studio: " + error_messages)
    
    @property
    def _transform_invoke_error_mapping(self) -> dict:
        """
        Get the mapping of invoke errors to model errors.
        
        :return: mapping dictionary
        """
        return {
            InvokeConnectionError: [
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError
            ],
            InvokeAuthorizationError: [requests.exceptions.InvalidHeader],
            InvokeBadRequestError: [
                requests.exceptions.HTTPError,
                requests.exceptions.InvalidURL,
            ],
            InvokeRateLimitError: [requests.exceptions.RetryError],
            InvokeServerUnavailableError: [
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ],
        }

    def _get_context_size(self, model: str, credentials: dict) -> int:
        """
        Get context size from credentials

        :param model: model name
        :param credentials: model credentials
        :return: context size
        """
        return int(credentials.get("context_size", 4096)) 