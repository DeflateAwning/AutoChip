# Python Modules
from abc import ABC, abstractmethod
import os
from loguru import logger

# AI Modules
import openai

from anthropic import Anthropic
from anthropic import AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT

import google.generativeai as palm

from conversation import Conversation

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import torch

# Modules from this project
from utils import find_verilog_modules


# Abstract Large Language Model
# Defines an interface for using different LLMs so we can easily swap them out
class AbstractLLM(ABC):
    """Abstract Large Language Model."""

    def __init__(self):
        pass

    @abstractmethod
    def generate(self, conversation: Conversation) -> str:
        """Generate a response based on the given conversation."""
        pass


class ChatGPT3p5(AbstractLLM):
    """ChatGPT Large Language Model."""

    def __init__(self):
        super().__init__()
        openai.api_key=os.environ['OPENAI_API_KEY']

    def generate(self, conversation: Conversation):
        messages = [{'role' : msg['role'], 'content' : msg['content']} for msg in conversation.get_messages()]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages = messages,
        )

        return response['choices'][0]['message']['content']

class ChatGPT4(AbstractLLM):
    """ChatGPT Large Language Model."""

    def __init__(self):
        super().__init__()
        openai.api_key=os.environ['OPENAI_API_KEY']

    def generate(self, conversation: Conversation):
        messages = [{'role' : msg['role'], 'content' : msg['content']} for msg in conversation.get_messages()]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages = messages,
        )

        return response['choices'][0]['message']['content']

class Claude(AbstractLLM):
    """Claude Large Language Model."""

    def __init__(self):
        super().__init__()
        self.anthropic = Anthropic(
            api_key=os.environ['ANTHROPIC_API_KEY'],
        )

    def generate(self, conversation: Conversation):
        prompt = ""
        for message in conversation.get_messages():
            if message['role'] == 'system' or message['role'] == 'user':
                prompt += f"\n\nHuman: {message['content']}"
            elif message['role'] == 'assistant':
                prompt += f"\n\nAssistant: {message['content']}"
        prompt += "\n\nAssistant:"


        completion = self.anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=3000,
            prompt=prompt,
        )

        #logger.info(prompt)
        #logger.info(completion.completion)
        return completion.completion

class PaLM(AbstractLLM):
    """PaLM Large Language Model."""

    def __init__(self):
        super().__init__()
        palm.configure(api_key=os.environ['PALM_API_KEY'])

    def generate(self, conversation: Conversation):

        context = None
        messages = []
        reply = ''

        for message in conversation.get_messages():
            if message['role'] == 'system':
                context = message['content']
            else:
                if message['role'] == 'user':
                    messages.append({'author': '0', 'content': message['content']})
                elif message['role'] == 'assistant':
                    messages.append({'author': '1', 'content': message['content']})

        response = palm.chat(context=context, messages=messages)
        #logger.info(response)
        return response.last

from transformers import CodeLlamaTokenizer, LlamaForCausalLM
class CodeLlama(AbstractLLM):
    """CodeLlama Large Language Model.
    Follow the setup instructions here: https://huggingface.co/welcome
    Install git-lfs:
        Option 1: https://github.com/git-lfs/git-lfs/issues/4134#issuecomment-1635204943
        Option 2: https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab

    Model list: https://huggingface.co/models?search=code_llama
    """

    def __init__(self, model_id="codellama/CodeLlama-13b-hf"):
        super().__init__()

        self.model_id = model_id

        # As of now, the following two commands download massive
        # model files to: ~/.cache/huggingface/hub/
        # Option 1: Use cache_dir kwarg to change this location.
        # Option 2: Use the DF_DATASETS_CACHE environment variable to change this location.
        #     $ export HF_DATASETS_CACHE="/path/to/another/directory"
        # TODO: do this download before starting the GPU instance

        code_llama_model_id = "codellama/CodeLlama-7b-Instruct-hf"

        self.tokenizer = CodeLlamaTokenizer.from_pretrained(code_llama_model_id)
        logger.info(f"Constructed tokenizer: {self.tokenizer}")

        self.model = LlamaForCausalLM.from_pretrained(
            code_llama_model_id,
            device_map="auto", torch_dtype = "auto")
        logger.info(f"Constructed model: {self.model}")
        assert isinstance(self.model, LlamaForCausalLM)

    def _format_prompt(self, conversation: Conversation) -> str:
        # Extract the system prompt, initial user prompt, and the most recent user prompt and answer.
        messages = conversation.get_messages()
        # each message has a 'role' and 'content' key
        # TODO: create a Message dataclass to enforce this structure

        prompt = ""


        user_message=""
        # system_prompt = ""
        # answer_message=""

        for message in messages:
            # Append system messages with the "<<SYS>>" tags
            if message['role'] == 'system':
                #system_prompt = message['content']
                prompt += f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n"
            # Append user messages with the "Human" prefix
            elif message['role'] == 'user':
                user_message = message['content']
                prompt += f"<s>[INST] {user_message.strip()} [/INST] "
            # Append assistant messages with the "Assistant" prefix wrapped with [INST] tags
            elif message['role'] == 'assistant':
                prompt += f"{message['content']}"
                #answer_message = message['content']

            #context = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message}"
            #prompt += f"<s>[INST] {context} [/INST] {answer_message}"

        logger.info(f"Running prompt:\n{prompt}")
        return prompt

    def generate(self, conversation: Conversation):

        # Prepare the prompt using the method we created
        prompt = self._format_prompt(conversation)

        tokenizer_inst = self.tokenizer(prompt, return_tensors="pt")
        logger.debug("Made tokenizer_inst")
        inputs = tokenizer_inst.to("cuda")
        logger.debug("Moved tokenizer_inst to cuda")

        assert isinstance(self.model, LlamaForCausalLM)
        output = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=3000,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        logger.debug("Generated output: self.model.generate(...)")

        # Move the output tensor to the CPU
        output = output[0].to("cpu")
        logger.debug("Moved output to CPU")

        # Decode the output to get the generated text
        decoded_output = self.tokenizer.decode(output)
        logger.debug("Decoded output")
        
        # Extract only the generated response
        response = decoded_output.split("[/INST]")[-1].strip()

        logger.info(f"RAW RECEIVED RESPONSE:\n{decoded_output}")

        #response = find_verilog_modules(decoded_output)[-1]

        logger.info('RESPONSE START')
        logger.info('\n'.join(find_verilog_modules(response)))
        logger.info('RESPONSE END')
        return response
