from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator(object):
    def __init__(
            self
    ):
        self._tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        self._model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")

    def __call__(
            self,
            text: str
    ) -> str:
        input_ids = self._tokenizer.encode(text, return_tensors="pt")
        outputs = self._model.generate(input_ids)
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
