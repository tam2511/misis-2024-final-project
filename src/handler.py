from typing import Dict, Optional

from src.translator import Translator
from src.generator import Generator as _Generator


class Handler(object):
    def __init__(
            self
    ):
        self._translator = Translator()
        self._generator = _Generator()

    def __call__(
            self,
            text: str,
            translator_kwargs: Optional[Dict] = None,
            generator_kwargs: Optional[Dict] = None,
    ) -> str:
        if translator_kwargs is None:
            translator_kwargs = dict()
        eng_text = self._translator(
            text=text,
            **translator_kwargs
        )

        if generator_kwargs is None:
            generator_kwargs = dict()
        return self._generator(
            text=eng_text,
            **generator_kwargs
        )
