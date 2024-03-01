from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.handler import Handler


class Prompt(BaseModel):
    text: str
    num_steps: int = 6
    negative_prompt: str = "bad quality, worse quality, low resolution"
    num_frames: int = 20


app = FastAPI()
handler = Handler()


@app.post("/generate_video")
def read_root(
        prompt: Prompt
):
    generator = handler(
        text=prompt.text,
        generator_kwargs=dict(
            num_steps=prompt.num_steps,
            negative_prompt=prompt.negative_prompt,
            num_frames=prompt.num_frames,
        )
    )
    return generator

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port='8889')
