from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Interface for language model backends.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
    ) -> str:
        pass