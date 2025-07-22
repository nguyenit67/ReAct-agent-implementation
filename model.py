import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda")
DEVICE_MAP = "cuda" if DEVICE_MAP == "cuda" and torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE_MAP}")


class Model:
    def __init__(self, model_id="google/medgemma-4b-it"):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # torch_dtype=torch.bfloat16,
            device_map=DEVICE_MAP,
        )
        self.processor = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def __call__(self, messages, max_new_tokens=500):
        """Run inference on provided messages and return the generated text."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self.model.device,
            # dtype=torch.bfloat16,
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
