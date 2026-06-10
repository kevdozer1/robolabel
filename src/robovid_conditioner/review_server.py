"""Browser-based calibration GUI (``robovid_conditioner review``).

A self-contained ``http.server`` single-page app — no Streamlit. It plays the
episode and lets you **scrub frame by frame**, watch the active subtask highlight
move with the playhead, and set a subtask boundary or a subgoal frame *from the
current frame*. Frames are served as exact per-index JPEGs straight from the
adapter, so boundary setting is frame-accurate and works for any source (LeRobot
or a directory of videos/frames) with only Pillow.

Your edits are written to the gold file's ``gold`` block via
:func:`robovid_conditioner.gold.update_episode_review`; the VLM ``auto`` labels
are never touched. The header shows live reliability as you review.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image

from .episode import Episode
from .gold import load_or_sync_gold, update_episode_review
from .reliability import reliability_report
from .schema import episode_records, list_episode_ids, read_annotations

_SEGMENT_COLORS = ["#4c78a8", "#59a14f", "#e8752a", "#b279a2", "#edc948", "#76b7b2"]


class ReviewSession:
    """Holds the annotations, the source frames, and the gold file for review."""

    def __init__(self, annotations_dir: str | Path, gold_file: str | Path, source=None):
        self.annotations_dir = Path(annotations_dir)
        self.gold_file = Path(gold_file)
        self.df = read_annotations(self.annotations_dir)
        self.episode_ids = list_episode_ids(self.df)
        load_or_sync_gold(self.annotations_dir, self.gold_file)
        # Map id -> Episode for frame serving (lazy: iterating does not decode).
        self.episodes: dict[str, Episode] = {}
        if source is not None:
            self.episodes = {ep.episode_id: ep for ep in source}
        self._frame_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._cache_cap = 800
        self._lock = threading.Lock()
        self._prefetch_token = 0

    # ---- state / payloads ------------------------------------------------ #
    def state(self) -> dict[str, Any]:
        gold = _read(self.gold_file)
        reviewed = {e["episode_id"] for e in gold["episodes"] if _is_reviewed(e)}
        report = reliability_report(self.gold_file)
        queue = []
        for ep in gold["episodes"]:
            auto_q = ep.get("auto", {}).get("metadata", {}).get("quality")
            gold_q = ep.get("gold", {}).get("metadata", {}).get("quality")
            queue.append({
                "episode_id": str(ep["episode_id"]),
                "task": ep.get("task") or "",
                "reviewed": str(ep["episode_id"]) in reviewed,
                "auto_score": auto_q,
                "gold_score": gold_q,
            })
        return {
            "annotations_dir": str(self.annotations_dir),
            "gold_file": str(self.gold_file.resolve()),
            "episode_count": len(queue),
            "reviewed_count": len(reviewed),
            "has_frames": bool(self.episodes),
            "quality_exact_agreement": report["quality_exact_agreement"],
            "quality_within_one_agreement": report["quality_within_one_agreement"],
            "boundary_iou": report["subtask_boundary_temporal_iou_mean"],
            "subgoal_agreement": report["subgoal_frame_agreement"],
            "queue": queue,
        }

    def episode_payload(self, episode_id: str) -> dict[str, Any]:
        rec = episode_records(self.df, episode_id)
        gold_entry = self._gold_entry(episode_id)
        auto = gold_entry.get("auto", {})
        gmeta = gold_entry.get("gold", {}).get("metadata", {})
        auto_meta = auto.get("metadata", {})
        segments = []
        for i, s in enumerate(rec["subtasks"]):
            segments.append({
                "segment_idx": int(s["segment_idx"]),
                "start_frame": int(s["start_frame"]),
                "end_frame": int(s["end_frame"]),
                "subtask_text": str(s["subtask_text"]),
                "color": _SEGMENT_COLORS[i % len(_SEGMENT_COLORS)],
            })
        num_frames = int(rec["num_frames"]) or (self.episodes[episode_id].num_frames if episode_id in self.episodes else 1)
        fps = float(self.episodes[episode_id].fps) if episode_id in self.episodes else 10.0
        payload = {
            "episode_id": episode_id,
            "task": rec["task"] or "",
            "num_frames": num_frames,
            "fps": fps,
            "has_frames": episode_id in self.episodes,
            "segments": segments,
            "subgoals": auto.get("subgoals", []),
            "auto_metadata": auto_meta,
            "review": {
                "gold_score": gmeta.get("quality"),
                "auto_score": auto_meta.get("quality"),
                "gold_mistake": gmeta.get("mistake"),
                "auto_mistake": auto_meta.get("mistake"),
                "reason": gmeta.get("reason") or "",
                "notes": gold_entry.get("review_notes", ""),
                "gold_subtasks": gold_entry.get("gold", {}).get("subtasks", []),
                "gold_subgoals": gold_entry.get("gold", {}).get("subgoals", []),
            },
            "prev_episode_id": self._neighbor(episode_id, -1),
            "next_episode_id": self._neighbor(episode_id, +1),
        }
        self._start_prefetch(episode_id, num_frames)
        return payload

    def _start_prefetch(self, episode_id: str, num_frames: int) -> None:
        """Warm the frame cache in the background (server-side, no browser flood).

        Decodes the episode's frames sequentially into the cache so scrubbing and
        playback hit warm cache. Cancels itself when a newer episode is opened.
        """
        if episode_id not in self.episodes:
            return
        with self._lock:
            self._prefetch_token += 1
            token = self._prefetch_token

        def work() -> None:
            for i in range(num_frames):
                if token != self._prefetch_token:
                    return  # a newer episode was opened
                try:
                    self.frame_jpeg(episode_id, i)
                except Exception:  # noqa: BLE001 - prefetch is best-effort
                    return

        threading.Thread(target=work, daemon=True).start()

    def save_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = str(payload.get("episode_id"))
        with self._lock:
            update_episode_review(
                self.gold_file, episode_id,
                quality=int(payload.get("score", 3)),
                mistake=bool(payload.get("mistake", False)),
                reason=str(payload.get("reason", "")),
                accept_auto_metadata=bool(payload.get("accept_auto_metadata", False)),
                gold_subtasks=payload.get("subtasks") or None,
                gold_subgoals=payload.get("subgoals") or None,
                review_notes=str(payload.get("notes", "")),
            )
        return {"saved": True, "episode_id": episode_id,
                "next_episode_id": self._next_unreviewed(episode_id) or self._neighbor(episode_id, +1),
                "state": self.state()}

    def frame_jpeg(self, episode_id: str, idx: int) -> bytes | None:
        if episode_id not in self.episodes:
            return None
        key = (episode_id, int(idx))
        # The whole decode is serialized: ThreadingHTTPServer handles each request
        # in its own thread, but the underlying video decoder (pyav, via LeRobot)
        # is NOT thread-safe — concurrent decodes return wrong/stuck frames. One
        # lock around cache-lookup + decode keeps it correct (a decode is ~ms and
        # results are cached, so a single reviewer scrubbing/playing is plenty fast).
        with self._lock:
            cached = self._frame_cache.get(key)
            if cached is not None:
                self._frame_cache.move_to_end(key)
                return cached
            arr = self.episodes[episode_id].frame(int(idx))
            buf = io.BytesIO()
            Image.fromarray(arr).convert("RGB").save(buf, format="JPEG", quality=82)
            data = buf.getvalue()
            self._frame_cache[key] = data
            while len(self._frame_cache) > self._cache_cap:
                self._frame_cache.popitem(last=False)
            return data

    # ---- helpers --------------------------------------------------------- #
    def _gold_entry(self, episode_id: str) -> dict[str, Any]:
        for e in _read(self.gold_file)["episodes"]:
            if str(e["episode_id"]) == str(episode_id):
                return e
        return {}

    def _neighbor(self, episode_id: str, step: int) -> str | None:
        ids = self.episode_ids
        if episode_id not in ids:
            return None
        return ids[(ids.index(episode_id) + step) % len(ids)]

    def _next_unreviewed(self, episode_id: str) -> str | None:
        gold = _read(self.gold_file)
        reviewed = {e["episode_id"] for e in gold["episodes"] if _is_reviewed(e)}
        ids = self.episode_ids
        if episode_id not in ids:
            return None
        start = ids.index(episode_id)
        for off in range(1, len(ids) + 1):
            cand = ids[(start + off) % len(ids)]
            if cand not in reviewed:
                return cand
        return None


def _is_reviewed(entry: dict[str, Any]) -> bool:
    return entry.get("gold", {}).get("metadata", {}).get("quality") is not None


def make_handler(session: ReviewSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "robovid_conditioner_review/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._html(INDEX_HTML)
                elif path == "/api/state":
                    self._json(session.state())
                elif path.startswith("/api/episode/"):
                    self._json(session.episode_payload(unquote(path.rsplit("/", 1)[-1])))
                elif path.startswith("/frame/"):
                    _, _, ep, idx = path.split("/", 3)
                    data = session.frame_jpeg(unquote(ep), int(idx))
                    if data is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                    else:
                        self._bytes(data, "image/jpeg")
                elif path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001 - defensive server guard
                self._json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                if urlparse(self.path).path == "/api/review":
                    self._json(session.save_review(payload))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=500)

        def log_message(self, *args: Any) -> None:
            return

        def _html(self, html: str) -> None:
            self._bytes(html.encode("utf-8"), "text/html; charset=utf-8")

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port from {start}")


def serve(session: ReviewSession, host: str = "127.0.0.1", port: int = 8787, open_browser: bool = True) -> None:
    port = _free_port(port)
    server = ThreadingHTTPServer((host, port), make_handler(session))
    url = f"http://{host}:{port}"
    print(f"robovid_conditioner review GUI: {url}")
    print(f"gold file: {session.gold_file.resolve()}")
    if not session.episodes:
        print("note: no --source given, so frames are not shown. Pass --source/--target to scrub the clip.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping review GUI.")
    finally:
        server.server_close()


def build_session(annotations: str, gold: str, source_kind: str | None, target: str | None) -> ReviewSession:
    source = None
    if source_kind and target:
        from .adapters import build_source
        source = build_source(source_kind, target)
    return ReviewSession(annotations, gold, source=source)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Browser calibration GUI.")
    p.add_argument("--annotations", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--source", choices=["lerobot", "directory"], default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    session = build_session(args.annotations, args.gold, args.source, args.target)
    serve(session, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>robovid_conditioner review</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2933;--muted:#64748b;--line:#d9dee7;--blue:#2563eb;--green:#16835b;}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
header{height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:5}
header h1{font-size:16px;margin:0}header .stats{display:flex;gap:16px;color:var(--muted);font-size:13px}
main{height:calc(100vh - 52px);display:grid;grid-template-columns:minmax(210px,250px) minmax(460px,1fr) minmax(340px,420px);gap:10px;padding:10px}
aside,section{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;min-height:0}
.queue{display:flex;flex-direction:column}.queue-toolbar{padding:8px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap}
.queue-list{overflow-y:auto;padding:6px}.qi{width:100%;text-align:left;border:1px solid transparent;background:transparent;padding:7px;border-radius:6px;cursor:pointer;color:var(--ink)}
.qi:hover{background:#f1f5f9}.qi.active{border-color:var(--blue);background:#eff6ff}.qi.reviewed .qid{color:var(--green)}
.qid{font-size:12px;font-weight:700}.qtask{font-size:12px;color:var(--muted);margin-top:2px;line-height:1.2}.qscore{font-size:12px;margin-top:3px}
.viewer{display:flex;flex-direction:column}.vtitle{padding:10px 12px 0}.vtitle h2{margin:0 0 4px;font-size:17px}.vtitle p{margin:0;color:var(--muted)}
.stage{padding:12px;display:flex;flex-direction:column;align-items:center}
#frame{width:100%;max-height:52vh;object-fit:contain;background:#0b1220;border-radius:6px}
.scrub{width:100%;margin-top:8px;display:flex;gap:8px;align-items:center}
.scrub input[type=range]{flex:1}.btn{background:#fff;color:var(--ink);border:1px solid var(--line);padding:7px 11px;border-radius:6px;cursor:pointer}
.btn.play{min-width:64px;font-weight:700}.fcount{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px;min-width:96px;text-align:right}
.timeline{padding:0 12px 8px}.active{min-height:40px;border-radius:6px;color:#fff;font-weight:700;padding:9px;margin-bottom:8px;display:flex;align-items:center}
.bar{display:flex;width:100%;height:34px;border:1px solid var(--line);border-radius:6px;overflow:hidden;cursor:pointer}
.seg{height:100%;color:#fff;font-size:11px;padding:4px;overflow:hidden;display:flex;align-items:center;justify-content:center;text-align:center;border-right:1px solid rgba(255,255,255,.3)}
.seg.on{outline:3px solid #111827;outline-offset:-3px}
.review{overflow-y:auto;padding:12px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.pill{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:3px 8px;font-size:12px;color:var(--muted)}
h3{margin:14px 0 6px;font-size:14px}.muted{color:var(--muted);font-size:12px}
.scores{display:grid;grid-template-columns:1fr;gap:6px;margin:8px 0}
.scores label{border:1px solid var(--line);border-radius:6px;padding:8px;cursor:pointer;display:flex;gap:8px;align-items:center}
.scores label:has(input:checked){border-color:var(--blue);background:#eff6ff}
.card{border:1px solid var(--line);border-radius:6px;padding:8px;margin:7px 0;background:#fff}.card h4{margin:0 0 6px;font-size:12px}
.grid3{display:grid;grid-template-columns:64px 64px 1fr;gap:6px;align-items:center;margin-top:5px}
.grid3 input,.sg input{width:100%;border:1px solid var(--line);border-radius:5px;padding:6px;min-width:0}
.sg{display:grid;grid-template-columns:84px 1fr;gap:6px;align-items:center;margin-top:5px}
textarea{width:100%;min-height:52px;resize:vertical;border:1px solid var(--line);border-radius:6px;padding:7px;margin:4px 0 8px}
button.primary{background:var(--blue);color:#fff;border:0;padding:11px;border-radius:6px;font-weight:700;cursor:pointer;width:100%}
.checks label{display:block;margin:7px 0;font-size:13px}
</style></head><body>
<header><h1>robovid_conditioner · calibration</h1><div class="stats" id="stats"></div></header>
<main>
 <aside class="queue">
  <div class="queue-toolbar"><button class="btn" onclick="go(-1)">◀ Prev</button><button class="btn" onclick="go(1)">Next ▶</button><button class="btn" onclick="nextUnreviewed()">Next unreviewed</button></div>
  <div class="queue-list" id="queue"></div></aside>
 <section class="viewer">
  <div class="vtitle"><h2 id="epid">Loading…</h2><p id="task"></p></div>
  <div class="stage"><img id="frame" alt="frame">
   <div class="scrub"><button class="btn play" id="play" onclick="togglePlay()">▶ Play</button>
    <input type="range" id="slider" min="0" max="0" value="0" oninput="onScrub()">
    <span class="fcount" id="fcount">frame 0 / 0</span></div></div>
  <div class="timeline"><div class="active" id="active">No active subtask</div><div class="bar" id="bar"></div></div>
 </section>
 <section class="review">
  <div class="row"><span class="pill" id="autoScore"></span><span class="pill" id="autoMistake"></span></div>
  <h3>Your quality score</h3><div class="scores" id="scores"></div>
  <div class="checks"><label><input type="checkbox" id="mistake"> Mistake visible</label>
   <label><input type="checkbox" id="acceptMeta"> Auto metadata looks right</label>
   <label><input type="checkbox" id="acceptSubtasks"> Accept all auto subtask boundaries</label>
   <label><input type="checkbox" id="acceptSubgoals"> Accept all auto subgoal frames</label></div>
  <h3>Subtask boundaries</h3><div class="muted">Scrub to a transition, then “set boundary here”. Editing a row unchecks accept.</div><div id="boundaryEditor"></div>
  <h3>Subgoal frames</h3><div class="muted">A subgoal is the representative frame for a subtask (usually its end).</div><div id="subgoalEditor"></div>
  <h3>Reason</h3><textarea id="reason"></textarea><h3>Notes</h3><textarea id="notes"></textarea>
  <button class="primary" onclick="save(true)">Save review and next  (Ctrl+Enter)</button>
  <div class="row" style="margin-top:8px"><button class="btn" onclick="save(false)">Save only</button></div>
 </section></main>
<script>
const SCORES={1:'1 · reject',2:'2 · weak',3:'3 · partial',4:'4 · keep',5:'5 · clean keep'};
let S=null,E=null,cur=null,frame=0,playing=false,timer=null;
async function j(u,o={}){const r=await fetch(u,o);const d=await r.json();if(!r.ok||d.error)throw new Error(d.error||r.status);return d;}
async function init(){S=await j('/api/state');cur=(S.queue.find(q=>!q.reviewed)||S.queue[0]||{}).episode_id;renderState();if(cur)await load(cur);}
function renderState(){document.getElementById('stats').innerHTML=
 `<span>${S.reviewed_count}/${S.episode_count} reviewed</span><span>exact ${f(S.quality_exact_agreement)}</span><span>±1 ${f(S.quality_within_one_agreement)}</span><span>IoU ${f(S.boundary_iou)}</span><span>subgoal ${f(S.subgoal_agreement)}</span>`;
 const q=document.getElementById('queue');q.innerHTML='';
 for(const it of S.queue){const b=document.createElement('button');b.className=`qi ${it.reviewed?'reviewed':''} ${it.episode_id===cur?'active':''}`;
  b.onclick=()=>load(it.episode_id);const sc=it.gold_score==null?`auto ${it.auto_score}`:`auto ${it.auto_score} → you ${it.gold_score}`;
  b.innerHTML=`<div class="qid">${it.reviewed?'✓':'○'} ${it.episode_id}</div><div class="qtask">${esc(it.task)}</div><div class="qscore">${sc}</div>`;q.appendChild(b);}}
async function load(id){stop();cur=id;E=await j('/api/episode/'+encodeURIComponent(id));renderState();
 document.getElementById('epid').textContent=E.episode_id;document.getElementById('task').textContent=E.task;
 const sl=document.getElementById('slider');sl.max=Math.max(0,E.num_frames-1);frame=0;sl.value=0;showFrame();
 document.getElementById('autoScore').textContent='Auto score: '+(E.review.auto_score??'n/a');
 document.getElementById('autoMistake').textContent='Auto mistake: '+(E.review.auto_mistake?'yes':'no');
 document.getElementById('mistake').checked=Boolean(E.review.gold_mistake??E.review.auto_mistake);
 document.getElementById('acceptMeta').checked=false;
 document.getElementById('reason').value=E.review.reason||'';document.getElementById('notes').value=E.review.notes||'';
 renderScores(E.review.gold_score||E.review.auto_score||3);renderBoundaries();renderSubgoals();renderBar();}
function showFrame(){if(!E.has_frames){document.getElementById('frame').alt='no --source given';return;}
 document.getElementById('frame').src=`/frame/${encodeURIComponent(E.episode_id)}/${frame}`;
 document.getElementById('fcount').textContent=`frame ${frame} / ${E.num_frames-1}`;highlight();}
function onScrub(){frame=Number(document.getElementById('slider').value);showFrame();}
function togglePlay(){playing?stop():play();}
function play(){if(!E||!E.has_frames)return;playing=true;document.getElementById('play').textContent='❚❚ Pause';
 const fps=Math.min(E.fps||10,20);timer=setInterval(()=>{if(frame>=E.num_frames-1){stop();return;}frame++;document.getElementById('slider').value=frame;showFrame();},1000/fps);}
function stop(){playing=false;if(timer)clearInterval(timer);timer=null;const p=document.getElementById('play');if(p)p.textContent='▶ Play';}
function seekFrame(fr){frame=Math.max(0,Math.min(E.num_frames-1,fr));document.getElementById('slider').value=frame;showFrame();}
function highlight(){if(!E)return;let a=(E.segments||[]).find(s=>frame>=s.start_frame&&frame<=s.end_frame)||E.segments[0];
 const box=document.getElementById('active');if(a){box.style.background=a.color;box.textContent=`frame ${frame}: ${a.subtask_text}`;}
 for(const d of document.querySelectorAll('.seg'))d.classList.toggle('on',a&&Number(d.dataset.i)===Number(a.segment_idx));}
function renderBar(){const bar=document.getElementById('bar');bar.innerHTML='';const tot=Math.max(1,E.num_frames);
 for(const s of E.segments||[]){const d=document.createElement('div');d.className='seg';d.dataset.i=s.segment_idx;
  d.style.width=Math.max(2,((s.end_frame-s.start_frame+1)/tot)*100)+'%';d.style.background=s.color;d.textContent=s.subtask_text;
  d.title=`${s.start_frame}-${s.end_frame}: ${s.subtask_text}`;d.onclick=()=>seekFrame(s.start_frame);bar.appendChild(d);}highlight();}
function renderScores(sel){const g=document.getElementById('scores');g.innerHTML='';for(const s of [1,2,3,4,5]){const l=document.createElement('label');
 l.innerHTML=`<input type="radio" name="score" value="${s}" ${s===Number(sel)?'checked':''}> ${SCORES[s]}`;g.appendChild(l);}}
function renderBoundaries(){const root=document.getElementById('boundaryEditor');root.innerHTML='';
 const gold=new Map((E.review.gold_subtasks||[]).map(x=>[Number(x.segment_idx),x]));
 for(const s of E.segments||[]){const g=gold.get(Number(s.segment_idx))||{};const acc=g.accept_auto===true;
  const st=g.start_frame??s.start_frame,en=g.end_frame??s.end_frame,tx=g.subtask_text||s.subtask_text||'';
  const c=document.createElement('div');c.className='card brow';c.dataset.i=s.segment_idx;
  c.innerHTML=`<h4>segment ${s.segment_idx}: ${esc(s.subtask_text)}</h4>
   <label><input type="checkbox" class="bacc" ${acc?'checked':''}> accept auto boundary</label>
   <div class="grid3"><input type="number" class="bstart" min="0" max="${E.num_frames-1}" value="${st}">
    <input type="number" class="bend" min="0" max="${E.num_frames-1}" value="${en}"><input type="text" class="btext" value="${escA(tx)}"></div>
   <div class="row" style="margin-top:6px"><button class="btn" type="button" onclick="setBound(${s.segment_idx})">⟱ set end = current frame</button></div>`;
  root.appendChild(c);}
 document.getElementById('acceptSubtasks').onchange=e=>{for(const cb of document.querySelectorAll('.bacc'))cb.checked=e.target.checked;};
 for(const i of document.querySelectorAll('.bstart,.bend,.btext'))i.addEventListener('input',()=>{i.closest('.brow').querySelector('.bacc').checked=false;document.getElementById('acceptSubtasks').checked=false;});}
function setBound(idx){const rows=[...document.querySelectorAll('.brow')].sort((a,b)=>a.dataset.i-b.dataset.i);
 const r=rows.find(x=>Number(x.dataset.i)===Number(idx));if(!r)return;r.querySelector('.bend').value=frame;r.querySelector('.bacc').checked=false;
 const nx=rows.find(x=>Number(x.dataset.i)===Number(idx)+1);if(nx){nx.querySelector('.bstart').value=Math.min(E.num_frames-1,frame+1);nx.querySelector('.bacc').checked=false;}
 document.getElementById('acceptSubtasks').checked=false;}
function renderSubgoals(){const root=document.getElementById('subgoalEditor');root.innerHTML='';
 const gold=new Map((E.review.gold_subgoals||[]).map(x=>[Number(x.segment_idx),x]));
 for(const sg of E.subgoals||[]){const g=gold.get(Number(sg.segment_idx))||{};const acc=g.accept_auto===true;const fr=g.frame_idx??sg.frame_idx??0;
  const c=document.createElement('div');c.className='card srow';c.dataset.i=sg.segment_idx;
  c.innerHTML=`<h4>segment ${sg.segment_idx}: auto frame ${sg.frame_idx??'n/a'}</h4>
   <label><input type="checkbox" class="sacc" ${acc?'checked':''}> accept auto subgoal frame</label>
   <div class="sg"><input type="number" class="sframe" min="0" max="${E.num_frames-1}" value="${fr}">
    <button class="btn" type="button" onclick="useCur(${sg.segment_idx})">use current frame</button></div>`;
  root.appendChild(c);}
 document.getElementById('acceptSubgoals').onchange=e=>{for(const cb of document.querySelectorAll('.sacc'))cb.checked=e.target.checked;};
 for(const i of document.querySelectorAll('.sframe'))i.addEventListener('input',()=>{i.closest('.srow').querySelector('.sacc').checked=false;document.getElementById('acceptSubgoals').checked=false;});}
function useCur(idx){const r=[...document.querySelectorAll('.srow')].find(x=>Number(x.dataset.i)===Number(idx));if(!r)return;
 r.querySelector('.sframe').value=frame;r.querySelector('.sacc').checked=false;}
async function save(adv){const score=Number(document.querySelector('input[name=score]:checked')?.value||3);
 const payload={episode_id:E.episode_id,score,mistake:document.getElementById('mistake').checked,
  accept_auto_metadata:document.getElementById('acceptMeta').checked,
  subtasks:[...document.querySelectorAll('.brow')].map(r=>({segment_idx:Number(r.dataset.i),accept_auto:r.querySelector('.bacc').checked,start_frame:Number(r.querySelector('.bstart').value),end_frame:Number(r.querySelector('.bend').value),subtask_text:r.querySelector('.btext').value})),
  subgoals:[...document.querySelectorAll('.srow')].map(r=>({segment_idx:Number(r.dataset.i),accept_auto:r.querySelector('.sacc').checked,frame_idx:Number(r.querySelector('.sframe').value)})),
  reason:document.getElementById('reason').value,notes:document.getElementById('notes').value};
 const res=await j('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 S=res.state;if(adv&&res.next_episode_id)await load(res.next_episode_id);else await load(E.episode_id);}
function go(d){const id=d<0?E.prev_episode_id:E.next_episode_id;if(id)load(id);}
function nextUnreviewed(){const i=S.queue.findIndex(q=>q.episode_id===cur);for(let k=1;k<=S.queue.length;k++){const it=S.queue[(i+k)%S.queue.length];if(!it.reviewed)return load(it.episode_id);}}
function f(v){return v==null?'n/a':Number(v).toFixed(3);}
function esc(t){return String(t||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function escA(t){return esc(t)}
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter')save(true);
 if(e.target.matches('textarea,input'))return;
 if(e.key==='ArrowRight'){e.preventDefault();seekFrame(frame+1);}if(e.key==='ArrowLeft'){e.preventDefault();seekFrame(frame-1);}
 if(e.key===' '){e.preventDefault();togglePlay();}});
init().catch(err=>{document.body.innerHTML='<pre style="padding:20px;color:#b91c1c">'+esc(err.message)+'</pre>';});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
