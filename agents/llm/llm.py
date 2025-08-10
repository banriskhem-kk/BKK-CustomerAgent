from pathlib import Path
from typing import Dict, List
import logging
import yaml  # Requires `pyyaml` package: pip install pyyaml
from ollama import chat, ChatResponse
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


class LLMClient:
    """A client for interacting with an LLM."""

    def __init__(
        self,
        model: str = "llama3.1:latest",    # for openai, "gpt-4o" or more
        prompts_file: str = "agents/llm/prompts/prompts.yaml",
    ):
        """Initialize the LLM client with a specified model and prompts directory.

        Args:
            model (str): The LLM model to use (default: llama3.1:latest)
            prompts_file (str): Path to the prompts YAML file
        """
        self.model = model
        self.prompts_file = prompts_file

        ## Only if you used Cloud based models, from OpenAi or OpenRouter or more.

        # self.client = OpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=os.getenv("OPENROUTER_API"),
        # )

        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load prompt templates from YAML file.

        Returns:
            Dict[str, str]: Dictionary of prompt templates keyed by name
        """
        prompts = {}
        try:
            with open(self.prompts_file, "r") as f:
                yaml_content = yaml.safe_load(f)

                if isinstance(yaml_content, dict):
                    # YAML contains key-value prompt definitions (your current format)
                    prompts = yaml_content
                elif isinstance(yaml_content, list):
                    # YAML contains a list of prompt objects
                    for prompt_obj in yaml_content:
                        name = prompt_obj.get("name")
                        text = prompt_obj.get("text")
                        if name and text:
                            prompts[name] = text
                else:
                    logger.warning(f"Unexpected YAML structure in {self.prompts_file}")

            logger.info(f"Loaded prompts: {list(prompts.keys())}")
            return prompts

        except FileNotFoundError:
            logger.error(f"Prompts file not found: {self.prompts_file}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            return {}

    def chat(self, prompt_name: str = None, message: str = "", prompt: str = None, **kwargs) -> str:  # type: ignore
        """Interact with the LLM using either a prompt template or direct prompt.

        Args:
            prompt_name (str, optional): Name of the prompt template to use
            message (str): User message to format into the prompt template
            prompt (str, optional): Direct prompt string (alternative to prompt_name)
            **kwargs: Additional variables to format into the prompt template

        Returns:
            str: Response content from the LLM
        """
        try:
            if prompt:
                # Use direct prompt string
                prompt_content = prompt
            elif prompt_name:
                # Get the prompt template or raise an error if not found
                if prompt_name not in self.prompts:
                    available_prompts = list(self.prompts.keys())
                    logger.error(
                        f"Prompt '{prompt_name}' not found. Available prompts: {available_prompts}"
                    )
                    return ""

                # Get the template
                template_content = self.prompts[prompt_name]

                # Create a mapping of variables (convert to uppercase for consistency)
                variables = {
                    "MESSAGE": message,
                    **{k.upper(): v for k, v in kwargs.items()},
                }

                # Replace {{VARIABLE}} placeholders
                prompt_content = template_content
                for var_name, var_value in variables.items():
                    placeholder = "{{" + var_name + "}}"
                    prompt_content = prompt_content.replace(placeholder, str(var_value))

                # Also support lowercase for backward compatibility
                variables_lower = {"message": message, **kwargs}
                for var_name, var_value in variables_lower.items():
                    placeholder = "{{" + var_name + "}}"
                    prompt_content = prompt_content.replace(placeholder, str(var_value))

            else:
                logger.error("Either prompt_name or prompt must be provided")
                return ""

            logger.debug(f"Formatted prompt: {prompt_content}")

            # Construct messages with a default system prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Follow the instructions in the user message carefully.",
                },
                {"role": "user", "content": prompt_content},
            ]

            ## Response calling method for OpenAI 
            
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=messages,  # type: ignore
            # )
            # return response.choices[0].message.content  # type: ignore


            ## Response calling method for Ollama

            response: ChatResponse = chat(
                model=self.model,
                messages=messages,
            )
            return response["message"]["content"]

        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            return ""

    def list_prompts(self) -> List[str]:
        """Return a list of available prompt names.

        Returns:
            List[str]: List of available prompt names
        """
        return list(self.prompts.keys())

    def get_prompt(self, prompt_name: str) -> str:
        """Get the raw prompt template by name.

        Args:
            prompt_name (str): Name of the prompt template

        Returns:
            str: Raw prompt template or empty string if not found
        """
        return self.prompts.get(prompt_name, "")


def get_llm_client(
    model: str = "llama3.1:latest",
    prompts_file: str = "agents/llm/prompts/prompts.yaml",
) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        model (str): The LLM model to use
        prompts_file (str): Path to the prompts YAML file

    Returns:
        LLMClient: Configured LLM client instance
    """
    return LLMClient(model, prompts_file)
