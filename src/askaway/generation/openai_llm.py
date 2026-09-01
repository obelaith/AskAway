import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from askaway.generation.llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
    ):
        self.client = OpenAI(
            api_key=os.environ.get(
                "OPENAI_API_KEY"
            )
        )

        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text