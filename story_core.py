import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).parent
STATIC = ROOT / "static"
LIVE_FILE = STATIC / "class.json"
LIVE_URL = "app/static/class.json"


def form_qr() -> str:
    import base64
    for name, mime in (("qr.png", "image/png"), ("qr.jpg", "image/jpeg"), ("qr.jpeg", "image/jpeg"), ("qr.webp", "image/webp")):
        f = ROOT / name
        if f.exists():
            return f"data:{mime};base64," + base64.b64encode(f.read_bytes()).decode()
    return ""


def build_story(model_text: str, live: bool) -> str:
    tpl = (ROOT / "story.html").read_text(encoding="utf-8")
    pred = (ROOT / "predictor.js").read_text(encoding="utf-8")
    head = ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>')
    return head + (tpl.replace("__MODEL__", model_text.replace("</", "<\\/"))
               .replace("__PREDICTOR__", pred)
               .replace("__LIVE_URL__", LIVE_URL if live else "")
               .replace("__FORM_QR__", form_qr())) + "</body></html>"


def sheet_csv_url(link: str):
    link = (link or "").strip()
    if not link:
        return None
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", link)
    if not m:
        return link if pathlib.Path(link).exists() or link.startswith("http") else None
    gid = re.search(r"[#&?]gid=(\d+)", link)
    return (f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv"
            + (f"&gid={gid.group(1)}" if gid else ""))


def read_sheet(csv_url: str) -> pd.DataFrame:
    if re.search(r"forms\.gle|/forms/", csv_url):
        raise ValueError("class_sheet.txt has the FORM link. Put the responses SHEET link there "
                         "(Responses tab > Link to Sheets, then copy the sheet's address).")
    if not csv_url.startswith("http"):
        return pd.read_csv(csv_url, dtype=str).fillna("")
    import io
    import requests
    r = requests.get(csv_url, timeout=10)
    if r.status_code in (401, 403) or "text/html" in r.headers.get("content-type", ""):
        raise ValueError("The sheet isn't shared. In the sheet: Share > General access > Anyone with the link > Viewer.")
    r.raise_for_status()
    text = r.content.decode("utf-8-sig")
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), dtype=str).fillna("")


def sync_responses(csv_url: str) -> int:
    df = read_sheet(csv_url)
    payload = {"headers": [str(c) for c in df.columns], "rows": df.values.tolist()}
    text = json.dumps(payload, ensure_ascii=False)
    STATIC.mkdir(exist_ok=True)
    if not LIVE_FILE.exists() or LIVE_FILE.read_text(encoding="utf-8") != text:
        LIVE_FILE.write_text(text, encoding="utf-8")
    return len(df)


def reset_live_file():
    STATIC.mkdir(exist_ok=True)
    LIVE_FILE.write_text(json.dumps({"headers": [], "rows": []}), encoding="utf-8")


def model_summary(model_text: str) -> dict:
    m = json.loads(model_text)
    return {
        "kernel": m["svm"]["kernel"], "support_vectors": len(m["svm"]["sv"]),
        "params": m["svm"].get("params", {}), "test_students": len(m["test"]),
        "svm": m["metrics"]["SVM"], "rival_name": m.get("rival_name"),
        "rival": m["metrics"].get("Rival") or m["metrics"].get("LogReg"),
    }
