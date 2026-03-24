#!/usr/bin/env python3
import json
import os
import time
import traceback
from pathlib import Path
from google import genai
from google.genai.errors import ClientError

def clean_json_text(raw: str) -> str:
    s = (raw or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def wait_active(client, file_obj, sleep_s=10, max_wait=300):
    """Wait until uploaded file becomes ACTIVE (or raise)."""
    start = time.time()
    state = getattr(file_obj, "state", None)
    state_name = getattr(state, "name", str(state))
    while state_name == "PROCESSING":
        if time.time() - start > max_wait:
            raise RuntimeError("File never became ACTIVE (timeout)")
        time.sleep(sleep_s)
        file_obj = client.files.get(name=file_obj.name)
        state = getattr(file_obj, "state", None)
        state_name = getattr(state, "name", str(state))
    if state_name != "ACTIVE":
        raise RuntimeError(f"File not ACTIVE (state={state_name})")
    return file_obj

def main():
    API_KEY = os.getenv("GEMINI_API_KEY")
    model_name = "gemini-2.0-flash"
    procedure_name = "laparoscopic cholecystectomy"
    plan_dir = Path("configs/window_plans")

    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=API_KEY)

    prompt_tmpl = """
You are watching a {PROCEDURE_NAME} surgery video.
The clip starts at time 0 seconds and lasts {CLIP_LEN} seconds.

First, understand what is happening in the clip as a surgeon would.
Then divide the clip into consecutive time segments and describe what happens in each segment.

Return ONLY valid JSON in the following format:
{
  "video_id": "{VIDEO_ID}",
  "window": "{WINDOW_NAME}",
  "segments": [
    { "start_sec": 0, "end_sec": 12, "description": "..." }
  ]
}

Guidelines:
- Use integer seconds and keep all times within 0 to {CLIP_LEN}.
- Write clear, descriptive explanations of the surgical actions.
- Create about 10–20 segments for a 5-minute clip (avoid many tiny segments).
- Mention tools and anatomy when visible.
- Avoid vague phrases like “view change” or “internal anatomy”.
- Prefer segment durations around 10–25 seconds, unless a clear action change happens.
- Do not guess the procedure type or add extra commentary.
- Do not mention brand names; use generic tool names only.
- No text outside the JSON.

Now produce the JSON segments for the clip.
""".strip()

    plan_files = sorted(plan_dir.glob("*.json"))
    print(f"[INFO] Found plans: {len(plan_files)}")

    for vi, plan_path in enumerate(plan_files, 1):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        video_id = plan["video_id"]
        windows = plan["windows"]

        win_dir = Path("data/windows") / video_id
        out_dir = Path("data/windows_captions_gemini/window_captions_1") / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== ({vi}/{len(plan_files)}) {video_id} windows={len(windows)} ===")

        for wi, w in enumerate(windows, 1):
            window_name = w["name"]
            clip_path = win_dir / f"{window_name}.mp4"
            out_path = out_dir / f"{window_name}.json"

            if out_path.exists():
                continue
            if not clip_path.exists():
                print(f"[MISS] {video_id} {window_name} missing clip")
                continue

            clip_len = int(w.get("end_sec", 0) - w.get("start_sec", 0))
            if clip_len <= 0:
                a, b = window_name.split("_")
                clip_len = int(b) - int(a)

            prompt = (
                prompt_tmpl
                .replace("{PROCEDURE_NAME}", procedure_name)
                .replace("{CLIP_LEN}", str(clip_len))
                .replace("{VIDEO_ID}", video_id)
                .replace("{WINDOW_NAME}", window_name)
            )

            file_obj = None
            raw = ""
            try:
                print(f"[UPLD] ({vi}/{len(plan_files)}) {video_id} {window_name}")
                uploaded = client.files.upload(file=str(clip_path))
                file_obj = wait_active(client, uploaded)

                print(f"[GEN ] ({vi}/{len(plan_files)}) {video_id} {window_name}")
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[file_obj, prompt],
                )

                raw = resp.text or ""
                clean = clean_json_text(raw)
                data = json.loads(clean)

                out_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                segs = len(data.get("segments", []))
                print(f"[OK  ] {video_id} {window_name} segs={segs}")

            except ClientError as e:
                print(f"[FAIL] {video_id} {window_name} ClientError: {repr(e)}")
                time.sleep(2)
            except json.JSONDecodeError as e:
                print(f"[FAIL] {video_id} {window_name} JSON parse error: {repr(e)}")
                try:
                    debug_path = out_dir / f"{window_name}_raw.txt"
                    debug_path.write_text(raw or repr(e), encoding="utf-8")
                    print(f"  [DEBUG] wrote raw output to {debug_path}")
                except Exception:
                    pass
            except Exception as e:
                print(f"[FAIL] {video_id} {window_name} err={repr(e)}")
                traceback.print_exc()
            finally:
                if file_obj is not None:
                    try:
                        client.files.delete(name=file_obj.name)
                        time.sleep(0.5)
                    except Exception as e_del:
                        print(f"[WARN] failed deleting remote file {getattr(file_obj, 'name', None)}: {repr(e_del)}")

    print("\n[DONE] All captioning attempted (can rerun to resume).")

if __name__ == "__main__":
    main()