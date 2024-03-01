from typing import Generator
import io

from PIL import Image
import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter


def image2bytes(
        image: Image
) -> bytes:
    out = io.BytesIO()
    image.save(out)
    return out.getvalue()

class Generator(object):
    def __init__(
            self
    ):
        adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=torch.float16)
        self._pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter,
                                                   torch_dtype=torch.float16)
        self._pipe.scheduler = LCMScheduler.from_config(self._pipe.scheduler.config, beta_schedule="linear")
        # self._pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors",
        #                        adapter_name="lcm-lora")
        self._pipe.set_adapters(["lcm-lora"], [0.8])

        self._pipe.enable_vae_slicing()
        self._pipe.enable_model_cpu_offload()

    def __call__(
            self,
            text: str,
            negative_prompt: str = "bad quality, worse quality, low resolution",
            num_frames: int = 16,
            num_inference_steps: int = 6
    ) -> Generator[bytes, None, None]:
        output = self._pipe(
            prompt=text,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=2.0,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(0),
        )
        frames = output.frames[0]
        for frame in frames:
            yield image2bytes(frame)
