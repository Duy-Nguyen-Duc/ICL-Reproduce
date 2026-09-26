"""Instruction-hint service: a generative VLM that writes the extra text for gr00t/rag/icl.py.

GR00T-N1.5's own VLM keeps only the first 12 LLM layers, so it cannot generate text. This
runs a separate VLM (default Qwen3-VL-30B-A3B-Instruct) in its own environment -- it needs
a newer transformers than the policy -- behind a stdlib HTTP endpoint:

  POST /hint  {"current_image": b64 png, "reference_image": b64 png,
               "reference_instruction": str, "instruction": str} -> {"hint": str}
"""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = (
    "You are helping a robot manipulation policy that was trained only on reference "
    "demonstrations.\n"
    "Image 1 is a reference demonstration frame for the trained task: \"{reference}\".\n"
    "Image 2 is the robot's current view. The scene may be perturbed (camera angle, "
    "lighting, background texture, object layout, robot start pose, sensor noise) and its "
    "instruction may be reworded or carry extra tags: \"{instruction}\".\n"
    "In ONE short imperative sentence (at most 20 words, lowercase, no punctuation other "
    "than commas), state the manipulation task the robot must do now, using the object "
    "names of the trained task. Output only that sentence."
)


def decode(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2").eval()

    @torch.inference_mode()
    def hint(req):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": decode(req["reference_image"])},
            {"type": "image", "image": decode(req["current_image"])},
            {"type": "text", "text": PROMPT.format(
                reference=req["reference_instruction"], instruction=req["instruction"])},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return " ".join(text.strip().strip(".").split())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                payload, code = {"hint": hint(body)}, 200
            except Exception as exc:  # report to the caller instead of dropping the socket
                payload, code = {"error": repr(exc)}, 500
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    print("Hint VLM is ready", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
