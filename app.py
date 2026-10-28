import json

import streamlit as st
import streamlit.components.v1 as components

from story_core import LIVE_FILE, ROOT, STATIC, build_story, reset_live_file, sheet_csv_url, sync_responses

st.set_page_config(page_title="On the Margin", page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stAppDeployButton, .stDeployButton, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
[data-testid="stAppViewContainer"], .stApp { background: #f6f1ff; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
iframe { display: block; width: 100% !important; height: 100vh !important; border: 0; }
</style>
""", unsafe_allow_html=True)

model_path = ROOT / "model.json"
if not model_path.exists():
    st.error("model.json is missing. Run export_for_story(...) in your notebook and put model.json next to app.py.")
    st.stop()
model_text = model_path.read_text(encoding="utf-8")

link_file = ROOT / "class_sheet.txt"
saved_link = ""
if link_file.exists():
    saved_link = next((l.strip() for l in link_file.read_text(encoding="utf-8").splitlines()
                       if l.strip() and not l.strip().startswith("#")), "")
if not saved_link:
    try:
        saved_link = str(st.secrets.get("class_sheet", "")).strip()
    except Exception:
        saved_link = ""
csv_url = sheet_csv_url(saved_link)
live = bool(csv_url)


_last = {"msg": None}


def _say(msg):
    if msg != _last["msg"]:
        _last["msg"] = msg
        print("[live]", msg, flush=True)


@st.fragment(run_every="2s")
def live_sync():
    try:
        n = sync_responses(csv_url)
        _say(f"OK, {n} responses in the sheet")
    except Exception as e:
        STATIC.mkdir(exist_ok=True)
        LIVE_FILE.write_text(json.dumps({"error": str(e)[:240]}), encoding="utf-8")
        _say(f"ERROR: {e}")


if live:
    print("[live] reading", csv_url, flush=True)
    live_sync()
else:
    print("[live] OFF: no sheet link in class_sheet.txt", flush=True)
    reset_live_file()

components.html(build_story(model_text, live), height=900, scrolling=False)
