from baseclasses.base_classes import BaseInferencer
import boto3
from typing import List, Dict, Any, Union, Tuple
import logging
from config.experimental_config import ExperimentalConfig, NShotPromptGuide
from core.inference.inference_factory import InferencerFactory
from util.boto3_utils import BedRockRetryHander
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Class for handling inference using Amazon Bedrock 
class BedrockInferencer(BaseInferencer):
    """Base class for all Bedrock models since they share the same invocation pattern"""

    # This part was added as part of the SageMaker implementation changes. 
    # Due to updates in the base class implementation, the invocation point was changed from 
    # '_initialize_client' to '__init__'.
    def __init__(self, model_id: str, experiment_config: ExperimentalConfig, region: str = 'us-east-1', role_arn: str = None):
        super().__init__(model_id, experiment_config, region, role_arn)
        self._initialize_client() 
    
    def _initialize_client(self) -> None:
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.region_name
        )

    def generate_prompt(self, experiment_config: ExperimentalConfig, default_prompt: str, user_query: str, context: List[Dict] = None) -> Tuple[str, List[Dict[str, Any]]]:
        # Get n_shot config values first to avoid repeated lookups
        n_shot_prompt_guide = experiment_config.n_shot_prompt_guide_obj
        n_shot_prompt = experiment_config.n_shot_prompts

        messages = []

        context_text = ""
        if context:
            context_text = self._format_context(context)
            if context_text:
                messages.append(self._prepare_conversation(role="user", message=context_text))

        # Input validation
        if n_shot_prompt < 0:
            raise ValueError("n_shot_prompt must be non-negative")
        
        # Get system prompt
        system_prompt = default_prompt if n_shot_prompt_guide is None or n_shot_prompt_guide.system_prompt is None else n_shot_prompt_guide.system_prompt
        
        base_prompt = n_shot_prompt_guide.user_prompt if n_shot_prompt_guide.user_prompt else ""
        messages.append(self._prepare_conversation(role="user", message=base_prompt))
        
        # Get examples
        examples = n_shot_prompt_guide.examples
        
        # Format examples
        selected_examples = (random.sample(examples, n_shot_prompt) 
                        if len(examples) > n_shot_prompt 
                        else examples)
        
        # Use string concatenation for example formatting
        for example in selected_examples:
                if 'example' in example:
                    messages.append(self._prepare_conversation(role="user", message=example['example']))
                elif 'question' in example and 'answer' in example:
                    messages.append(self._prepare_conversation(role="user", message=example['question']))
                    messages.append(self._prepare_conversation(role="assistant", message=example['answer']))
        
        logger.info(f"into {n_shot_prompt} shot prompt  with examples {len(selected_examples)}")

        # Add the current user prompt
        messages.append(self._prepare_conversation(role="user", message=user_query))
        
        return system_prompt, messages
     
    @BedRockRetryHander()
    def generate_text(self, user_query: str, default_prompt: str, context: List[Dict] = None, **kwargs) -> Tuple[Dict[Any, Any], str]:
        try:
            # Code to generate prompt considering the upload prompt config file
            system_prompt, messages = self.generate_prompt(self.experiment_config, default_prompt, user_query, context)

            inference_config = {
                "maxTokens": 512, 
                "temperature": self.experiment_config.temp_retrieval_llm, 
                "topP": 0.9
            }
            
            skip_system_param = self.model_id in ("amazon.titan-text-express-v1", "amazon.titan-text-lite-v1", "mistral.mistral-7b-instruct-v0:2")

            request_params = {
                "modelId": self.model_id,
                "messages": ([self._prepare_conversation(role="user", message=system_prompt)] if skip_system_param else []) + messages,
                "inferenceConfig": inference_config
            }
            
            # Add system parameter only for non-Titan-v1 models
            #TODO: Short-term fix, will be addressed using inheritence as part of refactoring
            if not skip_system_param:
                request_params["system"] = [{"text" : system_prompt}]
            
            response = self.client.converse(**request_params)
           
            metadata = {}
            if 'usage' in response:
                for key, value in response['usage'].items():
                    metadata[key] = value
            if 'metrics' in response:
                for key, value in response['metrics'].items():
                    metadata[key] = value
            return metadata, self._extract_response(response)
        except Exception as e:
            logger.error(f"Error generating text with Bedrock: {str(e)}")
            raise

    def _prepare_conversation(self, message: str, role: str):
        # Format message and role into a conversation
        if not message or not role:
            logger.error(f"Error in parsing message or role")
        conversation = {
                "role": role, 
                "content": [{"text" : message}]
            }
        
        return conversation

    def _format_context(self, context: List[Dict[str, str]]) -> str:
        """Format context documents into a single string."""
        context_text = "\n".join([
            f"Context {i+1}:\n{doc.get('text', '')}"
            for i, doc in enumerate(context)
        ])
        logger.debug(f"Formatted context text length: {len(context_text)}")
        context_text = "<context>\n" + context_text + "\n" +  "</context>\n"
        return context_text

    def _extract_response(self, response: Dict) -> str:
        """Extract text from Bedrock response."""
        response_text = response["output"]["message"]["content"][0]["text"]
        logger.info("Successfully generated response")
        logger.debug(f"Response length: {len(response_text)}")
        return response_text
    
    
model_list = ["mistral.mistral-7b-instruct-v0:2",
              "mistral.mistral-large-2402-v1:0",
              "us.meta.llama3-2-90b-instruct-v1:0",
              "us.meta.llama3-2-11b-instruct-v1:0",
              "us.meta.llama3-2-3b-instruct-v1:0",
              "us.meta.llama3-2-1b-instruct-v1:0",
              "cohere.command-r-v1:0",
              "cohere.command-r-plus-v1:0",
              "anthropic.claude-3-5-sonnet-20240620-v1:0",
              "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
              "us.anthropic.claude-3-5-haiku-20241022-v1:0",
              "amazon.titan-text-express-v1",
              "amazon.titan-text-lite-v1",
              "us.amazon.nova-lite-v1:0",
              "us.amazon.nova-micro-v1:0",
              "us.amazon.nova-pro-v1:0",
              "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
              ]

for model in model_list:
    InferencerFactory.register_inferencer('bedrock', model, BedrockInferencer)
    
    