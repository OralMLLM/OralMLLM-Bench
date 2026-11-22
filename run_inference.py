"""Zero-shot inference for OralMLLM-Bench.

Wang R, Zhou S, Wang J, Xie W, Che X. Cognitive evaluation of multimodal large
language models in dental radiography. J Dent 2026;107174.

Protocol:
  * every image-task query is an independent request containing the system-level
    instruction, the original image file and the task-specific prompt;
  * image files are sent unmodified as Base64 data URLs;
  * temperature is 0 where the endpoint accepts it; no other decoding parameters
    are set for API-based models;
  * only the first returned response is kept;
  * queries run one at a time; the wall-clock time of each query is recorded.

Usage:
  python run_inference.py --manifest manifest.csv --images-root images/ --out responses.jsonl
  python run_inference.py ... --models Qwen-VL-Plus GLM-4.6V
  python run_inference.py ... --dry-run          # print request bodies without sending them

manifest.csv columns: task_id, radiograph_id, image_path (relative to --images-root).
API keys are read from environment variables (see MODELS below).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Evaluated models.
MODELS = {
    "GPT-5.2": {"model": "gpt-5.2", "url_env": "AZURE_OPENAI_CHAT_URL", "key_env": "AZURE_OPENAI_API_KEY",
                "auth": "api-key"},  # private Azure OpenAI deployment: full chat-completions URL via env var
    "GLM-4.6V": {"model": "glm-4.6v", "url": "https://api.z.ai/api/paas/v4/chat/completions",
                 "key_env": "ZAI_API_KEY"},
    "Kimi K2.5": {"model": "kimi-k2.5", "url": "https://api.kimi.com/coding/v1/chat/completions",
                  "key_env": "KIMI_API_KEY"},
    "Qwen-VL-Plus": {"model": "qwen-vl-plus",
                     "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                     "key_env": "DASHSCOPE_API_KEY"},
    "MiniMax-VL-01": {"model": "MiniMax-VL-01", "url": "https://api.minimax.io/v1/chat/completions",
                      "key_env": "MINIMAX_API_KEY"},
    "LLaVA-1.6-7B": {"model": "llava-hf/llava-v1.6-vicuna-7b-hf", "local": True},
}

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".bmp": "image/bmp",
        ".tif": "image/tiff", ".tiff": "image/tiff", ".webp": "image/webp"}
RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def load_prompts(path: Path) -> tuple[str, dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["system_prompt"], {t["task_id"]: t["prompt"] for t in data["tasks"]}


def data_url(path: Path) -> str:
    mime = MIME.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported image type: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def request_body(model_id: str, system_prompt: str, prompt: str, image_url: str, temperature) -> dict:
    body = {"model": model_id, "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                     {"type": "text", "text": prompt}]},
    ]}
    if temperature is not None:
        body["temperature"] = temperature
    return body


class ApiModel:
    def __init__(self, name: str, spec: dict, max_retries: int = 5, timeout: float = 600):
        self.name, self.model_id = name, spec["model"]
        self.url = spec.get("url") or os.environ.get(spec.get("url_env", ""), "")
        if not self.url:
            raise SystemExit(f"{name}: set the environment variable {spec['url_env']}")
        key = os.environ.get(spec["key_env"], "")
        self.headers = {"Content-Type": "application/json",
                        **({"api-key": key} if spec.get("auth") == "api-key" else {"Authorization": f"Bearer {key}"})}
        self.max_retries, self.timeout = max_retries, timeout
        self.temperature = 0  # set to None once the endpoint rejects a temperature setting

    def _post(self, body: dict):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"), headers=self.headers,
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate(self, system_prompt: str, prompt: str, image: Path) -> dict:
        url = data_url(image)
        attempts, error = 0, None
        while attempts < self.max_retries:
            attempts += 1
            try:
                raw = self._post(request_body(self.model_id, system_prompt, prompt, url, self.temperature))
                choice = (raw.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                return {"response_text": content, "finish_reason": choice.get("finish_reason"),
                        "temperature": self.temperature if self.temperature is not None else "provider default",
                        "attempts": attempts, "error": None}
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                error = f"HTTP {e.code}: {detail}"
                if e.code == 400 and self.temperature is not None and "temperature" in detail.lower():
                    self.temperature = None  # endpoint does not accept a temperature setting
                    continue
                if e.code not in RETRYABLE:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                error = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** attempts))
        return {"response_text": "", "finish_reason": None, "temperature": None, "attempts": attempts,
                "error": error}


class LocalLlava:
    """LLaVA-1.6-7B via Hugging Face transformers; greedy decoding (temperature 0)."""

    def __init__(self, name: str, spec: dict, max_new_tokens: int):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        self.name, self.model_id, self.max_new_tokens = name, spec["model"], max_new_tokens
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_id, torch_dtype="auto",
                                                                       device_map="auto")

    def generate(self, system_prompt: str, prompt: str, image: Path) -> dict:
        from PIL import Image

        conversation = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        except Exception:  # chat template without a system role
            text = f"{system_prompt} USER: <image>\n{prompt} ASSISTANT:"
        inputs = self.processor(images=Image.open(image).convert("RGB"), text=text,
                                return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens)
        reply = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return {"response_text": reply, "finish_reason": None, "temperature": 0, "attempts": 1, "error": None}


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("error") is None:
                keys.add((r["model"], r["task_id"], r["radiograph_id"]))
    return keys


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Zero-shot inference for OralMLLM-Bench")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--max-new-tokens", type=int, default=512, help="LLaVA-1.6-7B only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    system_prompt, prompts = load_prompts(Path(a.prompts))
    with open(a.manifest, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    unknown = sorted({q["task_id"] for q in queries} - set(prompts))
    if unknown:
        raise SystemExit(f"unknown task_id(s) in manifest: {unknown}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    for name in a.models:
        spec = MODELS[name]
        if a.dry_run:
            for q in queries[:1]:
                shown = ({"local": True, "system_prompt": system_prompt, "prompt": prompts[q["task_id"]]}
                         if spec.get("local") else
                         request_body(spec["model"], system_prompt, prompts[q["task_id"]], "data:<image>", 0))
                print(json.dumps({"model": name, "task_id": q["task_id"], "request": shown}, ensure_ascii=False))
            continue
        client = LocalLlava(name, spec, a.max_new_tokens) if spec.get("local") else ApiModel(name, spec)
        for q in queries:
            if (name, q["task_id"], q["radiograph_id"]) in done:
                continue
            started = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            result = client.generate(system_prompt, prompts[q["task_id"]], Path(a.images_root) / q["image_path"])
            record = {"model": name, "model_id": spec["model"], "task_id": q["task_id"],
                      "radiograph_id": q["radiograph_id"], "image_path": q["image_path"], **result,
                      "started_at_utc": started, "elapsed_s": round(time.perf_counter() - t0, 3)}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
