import os
from openai import OpenAI
from dotenv import load_dotenv

from model import Model


load_dotenv()

# token = os.getenv("HF_TOKEN")
# endpoint = "https://router.huggingface.co/featherless-ai/v1"
model_name = "google/medgemma-4b-it"


# client = OpenAI(
#     base_url=endpoint,
#     api_key=token,
# )


def format_message(role, content, image=None, multimodal=False):
    """
    Format a message for the chat model, including optional image data.

    Args:
        text (str): The text content of the message.
        image (str, optional): Base64 encoded image data. Defaults to None.

    Returns:
        dict: Formatted message dictionary.
    """
    if image:
        content = [{"type": "text", "text": content}, {"type": "image", "image": image}]
    elif multimodal:
        content = [{"type": "text", "text": content}]

    return {
        "role": role,
        "content": content,
    }


class Chat:
    """
    Wrapper class for OpenAI's Chat API. Designed by Simon, extended by NguyenIT67.
    This class manages the chat session, including system prompts and message history.
    """

    def __init__(self, system="", model=model_name):
        self.system = system
        self.model = Model(model_id=model)
        self.messages = []
        if self.system:
            self.messages.append(format_message("system", self.system))
        if isinstance(self.model, Model):
            endpoint = "local machine"
        print(f"Using model {model} on {endpoint}")

    def __call__(self, message):
        self.messages.append(format_message("user", message))
        result = self.execute()
        self.messages.append(format_message("assistant", result))
        return result

    def execute(self):
        response = self.model(messages=self.messages)
        # response = response.choices[0].message.content
        return response

    def clear(self):
        self.messages = [format_message("system", self.system)] if self.system else []
