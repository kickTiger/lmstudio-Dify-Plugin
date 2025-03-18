import json
import logging
import re
from collections.abc import Generator
from typing import Any, Optional, Union, cast
from urllib.parse import urljoin

import requests
import lmstudio
from lmstudio import LMStudioError

from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

logger = logging.getLogger(__name__)


class LmstudioLargeLanguageModel(LargeLanguageModel):
    """
    Model class for lmstudio large language model.
    """

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
        elif isinstance(error, LMStudioError):
            # Handle specific LM Studio errors
            if "Unauthorized" in error_messages or "API key" in error_messages:
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
            InvokeAuthorizationError: [],
            InvokeBadRequestError: [],
            InvokeRateLimitError: [],
            InvokeServerUnavailableError: [],
        }

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        return self._generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )
   
    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:
        """
        model_mode = self.get_model_mode(model, credentials)
        # Get approximate token count using GPT-2 tokenizer
        if model_mode == LLMMode.CHAT:
            return self._num_tokens_from_messages(prompt_messages)
        else:
            first_prompt_message = prompt_messages[0]
            if isinstance(first_prompt_message.content, str):
                text = first_prompt_message.content
            elif isinstance(first_prompt_message.content, list):
                text = ""
                for message_content in first_prompt_message.content:
                    if message_content.type == PromptMessageContentType.TEXT:
                        message_content = cast(
                            TextPromptMessageContent, message_content
                        )
                        text = message_content.data
                        break
            return self._get_num_tokens_by_gpt2(text)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            # Create a simple prompt to test the credentials
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
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Generate response using LM Studio API

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for function calling
        :param stop: stop words
        :param stream: whether to stream the response
        :param user: user identifier
        :return: LLM result or generator
        """
        try:
            # Set up base URL
            base_url = credentials.get("base_url", "")
            if not base_url.endswith("/"):
                base_url += "/"
                
            # Prepare headers and parameters
            params = {}
            if model_parameters and isinstance(model_parameters, dict):
                params.update(model_parameters)
            
            if stop:
                params["stop"] = stop
                
            # Get completion mode
            mode = credentials.get("mode", "chat")
            
            # Configure OpenAI-compatible client for LM Studio
            from openai import OpenAI
            client = OpenAI(base_url=f"{base_url}v1", api_key="lm-studio")
            
            if mode == "chat":
                # Process chat messages
                messages = self._convert_prompt_messages_to_chat_messages(prompt_messages)
                
                # Prepare tools for function calling if available
                tool_params = {}
                if tools and len(tools) > 0:
                    formatted_tools = []
                    for tool in tools:
                        formatted_tools.append({
                            "type": "function",
                            "function": tool.model_dump()
                        })
                    tool_params["tools"] = formatted_tools
                
                # Call Chat API
                if stream:
                    response_stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stream=True,
                        **params,
                        **tool_params
                    )
                    
                    # Return a generator for streaming
                    return self._process_chat_stream(
                        response_stream=response_stream,
                        model=model,
                        credentials=credentials,
                        prompt_messages=prompt_messages
                    )
                else:
                    # Call without streaming
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **params,
                        **tool_params
                    )
                    
                    # Process the complete response
                    return self._process_chat_response(
                        response=response,
                        model=model,
                        credentials=credentials,
                        prompt_messages=prompt_messages
                    )
            else:
                # Completion mode
                first_message = prompt_messages[0]
                if isinstance(first_message, UserPromptMessage):
                    prompt = self._get_prompt_text(first_message)
                    
                    if stream:
                        response_stream = client.completions.create(
                            model=model,
                            prompt=prompt,
                            stream=True,
                            **params
                        )
                        
                        return self._process_completion_stream(
                            response_stream=response_stream,
                            model=model,
                            credentials=credentials,
                            prompt_messages=prompt_messages
                        )
                    else:
                        response = client.completions.create(
                            model=model,
                            prompt=prompt,
                            **params
                        )
                        
                        return self._process_completion_response(
                            response=response,
                            model=model,
                            credentials=credentials,
                            prompt_messages=prompt_messages
                        )
                else:
                    raise ValueError("Completion mode requires a UserPromptMessage")
                    
        except Exception as e:
            raise self._invoke_error_mapping(e)
            
    def _convert_prompt_messages_to_chat_messages(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        """
        Convert Dify prompt messages to LM Studio chat messages format
        """
        messages = []
        
        for message in prompt_messages:
            if isinstance(message, SystemPromptMessage):
                messages.append({"role": "system", "content": message.content})
            elif isinstance(message, UserPromptMessage):
                if isinstance(message.content, str):
                    messages.append({"role": "user", "content": message.content})
                elif isinstance(message.content, list):
                    # Process content list (text and images)
                    content_parts = []
                    for content_item in message.content:
                        if content_item.type == PromptMessageContentType.TEXT:
                            content_item = cast(TextPromptMessageContent, content_item)
                            content_parts.append({"type": "text", "text": content_item.data})
                        elif content_item.type == PromptMessageContentType.IMAGE:
                            content_item = cast(ImagePromptMessageContent, content_item)
                            content_parts.append({
                                "type": "image_url", 
                                "image_url": {"url": content_item.data}
                            })
                    messages.append({"role": "user", "content": content_parts})
            elif isinstance(message, AssistantPromptMessage):
                if message.tool_calls and len(message.tool_calls) > 0:
                    tool_calls = []
                    for tool_call in message.tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                    messages.append({
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": tool_calls
                    })
                else:
                    messages.append({"role": "assistant", "content": message.content or ""})
            elif isinstance(message, ToolPromptMessage):
                messages.append({
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content
                })
                
        return messages
        
    def _get_prompt_text(self, message: UserPromptMessage) -> str:
        """Extract text from user prompt message"""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            for content_item in message.content:
                if content_item.type == PromptMessageContentType.TEXT:
                    content_item = cast(TextPromptMessageContent, content_item)
                    return content_item.data
        return ""
        
    def _process_chat_response(self, response, model: str, credentials: dict, 
                              prompt_messages: list[PromptMessage]) -> LLMResult:
        """Process a complete chat API response"""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            # Create assistant message
            assistant_message = AssistantPromptMessage(content=message.content)
            
            # Add tool calls if present
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        AssistantPromptMessage.ToolCall(
                            id=tool_call.id,
                            type="function",
                            function=AssistantPromptMessage.ToolCallFunction(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments
                            )
                        )
                    )
                assistant_message.tool_calls = tool_calls
            
            # Create LLM result
            return LLMResult(
                model=model,
                message=assistant_message,
                prompt_messages=prompt_messages,
                usage=self._calculate_usage_from_response(response),
                finish_reason=choice.finish_reason
            )
        else:
            # Empty response
            return LLMResult(
                model=model,
                message=AssistantPromptMessage(content=""),
                prompt_messages=prompt_messages,
                usage=None,
                finish_reason="incomplete"
            )
            
    def _process_completion_response(self, response, model: str, credentials: dict, 
                                   prompt_messages: list[PromptMessage]) -> LLMResult:
        """Process a complete completion API response"""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            
            # Create assistant message with text
            assistant_message = AssistantPromptMessage(content=choice.text)
            
            # Create LLM result
            return LLMResult(
                model=model,
                message=assistant_message,
                prompt_messages=prompt_messages,
                usage=self._calculate_usage_from_response(response),
                finish_reason=choice.finish_reason
            )
        else:
            # Empty response
            return LLMResult(
                model=model,
                message=AssistantPromptMessage(content=""),
                prompt_messages=prompt_messages,
                usage=None,
                finish_reason="incomplete"
            )
            
    def _process_chat_stream(self, response_stream, model: str, credentials: dict, 
                           prompt_messages: list[PromptMessage]) -> Generator:
        """Process a streaming chat API response"""
        # Initialize id and assistant message for streaming
        chunk_id = 0
        assistant_message = AssistantPromptMessage(content="")
        
        # Generate streaming chunks
        for chunk in response_stream:
            chunk_id += 1
            
            # Check if chunk has choices
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Handle content delta
                if hasattr(delta, 'content') and delta.content:
                    # Update assistant message
                    assistant_message.content = (assistant_message.content or "") + delta.content
                    
                    # Create delta for this chunk
                    delta_obj = LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=delta.content),
                        content=delta.content,
                        tool_calls=None
                    )
                    
                    # Create chunk to yield
                    chunk_obj = LLMResultChunk(
                        id=str(chunk_id),
                        model=model,
                        prompt_messages=prompt_messages,
                        object="chat.completion.chunk",
                        delta=delta_obj,
                        finish_reason=choice.finish_reason,
                        usage=None
                    )
                    
                    yield chunk_obj
                
                # Handle tool call deltas
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    # Process tool call deltas
                    if not assistant_message.tool_calls:
                        assistant_message.tool_calls = []
                    
                    for tool_call_delta in delta.tool_calls:
                        # Find or create tool call
                        tool_call = None
                        for tc in assistant_message.tool_calls:
                            if tc.id == tool_call_delta.id:
                                tool_call = tc
                                break
                        
                        if not tool_call:
                            tool_call = AssistantPromptMessage.ToolCall(
                                id=tool_call_delta.id,
                                type="function",
                                function=AssistantPromptMessage.ToolCallFunction(
                                    name="",
                                    arguments=""
                                )
                            )
                            assistant_message.tool_calls.append(tool_call)
                        
                        # Update tool call with delta
                        if hasattr(tool_call_delta, 'function'):
                            fn_delta = tool_call_delta.function
                            if hasattr(fn_delta, 'name') and fn_delta.name:
                                tool_call.function.name = fn_delta.name
                            if hasattr(fn_delta, 'arguments') and fn_delta.arguments:
                                tool_call.function.arguments = (tool_call.function.arguments or "") + fn_delta.arguments
                    
                    # Create delta for this chunk with tool calls
                    tool_call_deltas = []
                    for tc_delta in delta.tool_calls:
                        tc_function = None
                        if hasattr(tc_delta, 'function') and tc_delta.function:
                            fn = tc_delta.function
                            tc_function = AssistantPromptMessage.ToolCallFunction(
                                name=fn.name if hasattr(fn, 'name') else "",
                                arguments=fn.arguments if hasattr(fn, 'arguments') else ""
                            )
                        
                        tool_call_deltas.append(
                            AssistantPromptMessage.ToolCall(
                                id=tc_delta.id,
                                type="function",
                                function=tc_function
                            )
                        )
                    
                    delta_obj = LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(tool_calls=tool_call_deltas),
                        content=None,
                        tool_calls=tool_call_deltas
                    )
                    
                    # Create chunk to yield
                    chunk_obj = LLMResultChunk(
                        id=str(chunk_id),
                        model=model,
                        prompt_messages=prompt_messages,
                        object="chat.completion.chunk",
                        delta=delta_obj,
                        finish_reason=choice.finish_reason,
                        usage=None
                    )
                    
                    yield chunk_obj
                
                # Final chunk with finish reason if ending
                if choice.finish_reason:
                    chunk_obj = LLMResultChunk(
                        id=str(chunk_id),
                        model=model,
                        prompt_messages=prompt_messages,
                        object="chat.completion.chunk",
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=""),
                            finish_reason=choice.finish_reason
                        ),
                        finish_reason=choice.finish_reason,
                        usage=None
                    )
                    
                    yield chunk_obj
                    
    def _process_completion_stream(self, response_stream, model: str, credentials: dict, 
                                 prompt_messages: list[PromptMessage]) -> Generator:
        """Process a streaming completion API response"""
        # Initialize id and content for streaming
        chunk_id = 0
        content = ""
        
        # Generate streaming chunks
        for chunk in response_stream:
            chunk_id += 1
            
            # Check if chunk has choices
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                # Handle text delta
                if hasattr(choice, 'text') and choice.text:
                    # Update content
                    content += choice.text
                    
                    # Create delta for this chunk
                    delta_obj = LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=choice.text),
                        content=choice.text,
                        tool_calls=None
                    )
                    
                    # Create chunk to yield
                    chunk_obj = LLMResultChunk(
                        id=str(chunk_id),
                        model=model,
                        prompt_messages=prompt_messages,
                        object="completion.chunk",
                        delta=delta_obj,
                        finish_reason=choice.finish_reason,
                        usage=None
                    )
                    
                    yield chunk_obj
                
                # Final chunk with finish reason if ending
                if choice.finish_reason:
                    chunk_obj = LLMResultChunk(
                        id=str(chunk_id),
                        model=model,
                        prompt_messages=prompt_messages,
                        object="completion.chunk",
                        delta=LLMResultChunkDelta(
                            index=0,
                            message=AssistantPromptMessage(content=""),
                            finish_reason=choice.finish_reason
                        ),
                        finish_reason=choice.finish_reason,
                        usage=None
                    )
                    
                    yield chunk_obj
                    
    def _calculate_usage_from_response(self, response) -> dict:
        """Extract usage information from response if available"""
        try:
            if hasattr(response, 'usage'):
                usage = response.usage
                if callable(usage):
                    # 如果usage是一个函数而不是一个对象，直接返回空的使用统计
                    return {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                    
                return {
                    "prompt_tokens": usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 0,
                    "completion_tokens": usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0,
                    "total_tokens": usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                }
        except (AttributeError, TypeError) as e:
            # 捕获任何属性错误或类型错误，处理可能的边缘情况
            pass
            
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
    def _num_tokens_from_messages(self, messages: list[PromptMessage]) -> int:
        """
        Approximate token count for chat messages using GPT-2 tokenizer
        """
        total_tokens = 0
        for message in messages:
            if isinstance(message.content, str):
                total_tokens += self._get_num_tokens_by_gpt2(message.content)
            elif isinstance(message.content, list):
                for item in message.content:
                    if item.type == PromptMessageContentType.TEXT:
                        item = cast(TextPromptMessageContent, item)
                        total_tokens += self._get_num_tokens_by_gpt2(item.data)
            # Add overhead for message formatting
            total_tokens += 4  # approximate overhead per message
        return total_tokens
        
    def get_model_mode(self, model: str, credentials: dict) -> LLMMode:
        """Get the mode of the model (chat or completion)"""
        return LLMMode.value_of(credentials.get("mode", "chat"))
        
    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        Return model schema for customizable model

        :param model: model name
        :param credentials: credentials

        :return: model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
            parameter_rules=[],
        )

        return entity
