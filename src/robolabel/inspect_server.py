"""``robolabel inspect`` — the verification viewer (a judgment instrument, not an editor).

Loads a prebuilt inspect payload (``inspect_data.json`` from
``scripts/build_inspect_data.py``) and serves a single-page app over stdlib
``http.server`` (no extra deps, runs from a fresh venv with one command). It shows,
per episode:

* a **multi-track boundary timeline** — gold + each strategy/baseline, color-coded,
  phase labels on segments, the active segment following the playhead;
* **evidence-verification mode** — every grounded evidence string next to a thumbnail
  of its cited frame; one click jumps playback there (you grade each true/false);
* a **per-episode metric panel** — segment IoU, boundary precision/recall@±5, MAE, gate
  flags, cost, quality vs gold;
* **sort/filter** — worst-IoU first, failure-band, largest disagreement, gate-flagged.

In ``blind`` mode (the fresh-dataset trial) the track names are hidden, episodes are
shuffled, and a grading panel records per-boundary accept/reject, per-phase right/wrong,
per-evidence true/false, and an overall verdict to a grades file (tallied into
``FRESH_TRIAL_REPORT.md`` by ``robolabel trial-report``).
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


class InspectSession:
    """Holds the inspect payload, the frame source, and (blind mode) the grades file."""

    def __init__(self, data_path: str | Path, source=None, grades_path: str | Path | None = None):
        self.data = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.episodes: dict[str, Any] = {ep.episode_id: ep for ep in source} if source is not None else {}
        self.grades_path = Path(grades_path) if grades_path else None
        self._frame_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._cache_cap = 1000
        self._lock = threading.Lock()
        self._prefetch_token = 0
        # index episodes by id for fast payload lookup
        self._by_id = {e["episode_id"]: e for e in self.data["episodes"]}

    def state(self) -> dict[str, Any]:
        eps = []
        for e in self.data["episodes"]:
            eps.append({
                "episode_id": e["episode_id"], "task": e.get("task", ""),
                "sort_iou": e.get("sort_iou", 1.0), "n_flags": e.get("n_flags", 0),
                "graded": self._is_graded(e["episode_id"]),
            })
        return {
            "dataset": self.data.get("dataset", ""),
            "source_kind": self.data.get("source_kind", ""),
            "track_order": self.data["track_order"],
            "track_colors": self.data["track_colors"],
            "blind": self.data.get("blind", False),
            "has_frames": bool(self.episodes),
            "episodes": eps,
            "graded_count": sum(1 for e in eps if e["graded"]),
        }

    def episode_payload(self, episode_id: str) -> dict[str, Any]:
        e = dict(self._by_id.get(episode_id, {}))
        frame_ep = e.get("frame_ep", episode_id)  # blind items serve frames by the real episode
        if frame_ep in self.episodes:
            self._start_prefetch(frame_ep, int(e.get("num_frames", 0)))
        e["existing_grade"] = self._load_grade(episode_id)
        return e

    # ---- blind-mode grading --------------------------------------------- #
    def save_grade(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.grades_path:
            return {"saved": False, "error": "no --grades file configured"}
        with self._lock:
            grades = self._read_grades()
            grades[str(payload.get("episode_id"))] = payload
            self.grades_path.parent.mkdir(parents=True, exist_ok=True)
            self.grades_path.write_text(json.dumps(grades, indent=2), encoding="utf-8")
        return {"saved": True, "graded_count": len(grades)}

    def _read_grades(self) -> dict[str, Any]:
        if self.grades_path and self.grades_path.exists():
            try:
                return json.loads(self.grades_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _load_grade(self, episode_id: str):
        return self._read_grades().get(str(episode_id))

    def _is_graded(self, episode_id: str) -> bool:
        return str(episode_id) in self._read_grades()

    # ---- frame serving (same model as review_server) -------------------- #
    def _start_prefetch(self, episode_id: str, num_frames: int) -> None:
        if episode_id not in self.episodes:
            return
        with self._lock:
            self._prefetch_token += 1
            token = self._prefetch_token

        def work() -> None:
            for i in range(num_frames):
                if token != self._prefetch_token:
                    return
                try:
                    self.frame_jpeg(episode_id, i)
                except Exception:  # noqa: BLE001 - best-effort
                    return

        threading.Thread(target=work, daemon=True).start()

    def frame_jpeg(self, episode_id: str, idx: int) -> bytes | None:
        if episode_id not in self.episodes:
            return None
        key = (episode_id, int(idx))
        with self._lock:  # decoder (pyav) is not thread-safe; serialize decode+cache
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


def make_handler(session: InspectSession):
    class Handler(BaseHTTPRequestHandler):
        server_version = "robolabel_inspect/1.0"

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
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
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                if urlparse(self.path).path == "/api/grade":
                    self._json(session.save_grade(payload))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": str(exc)}, status=500)

        def log_message(self, *args: Any) -> None:
            return

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

        def _bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port from {start}")


def serve(session: InspectSession, host: str = "127.0.0.1", port: int = 8799, open_browser: bool = True) -> None:
    port = _free_port(port)
    server = ThreadingHTTPServer((host, port), make_handler(session))
    url = f"http://{host}:{port}"
    mode = "BLIND TRIAL" if session.data.get("blind") else "review"
    print(f"robolabel inspect ({mode}): {url}")
    if not session.episodes:
        print("note: no --source given, so frames are not shown. Pass --source/--target to scrub clips.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping inspect viewer.")
    finally:
        server.server_close()


def parse_episodes(spec: str | None) -> list[int] | None:
    """Parse an episode spec like "0-7" or "0,2,5" into a list of indices (None = all).

    Use a contiguous range (e.g. "0-7") matching how the data was annotated — frame
    indices are global, so a non-contiguous subset would misalign the served frames.
    """
    if not spec:
        return None
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out or None


def build_session(data: str, source_kind: str | None, target: str | None,
                  grades: str | None, camera_key: str | None = None,
                  episodes: str | None = None) -> InspectSession:
    source = None
    if source_kind and target:
        from .adapters import build_source
        kwargs: dict = {}
        if source_kind == "lerobot" and camera_key:
            kwargs["camera_key"] = camera_key
        eps = parse_episodes(episodes)
        if eps is not None:
            kwargs["episodes"] = eps
        source = build_source(source_kind, target, **kwargs)
    return InspectSession(data, source=source, grades_path=grades)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="robolabel inspect — verification viewer.")
    p.add_argument("--data", required=True, help="inspect_data.json from build_inspect_data.py")
    p.add_argument("--source", choices=["lerobot", "directory"], default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--camera-key", default=None)
    p.add_argument("--episodes", default=None,
                   help='limit the loaded source to these episodes, e.g. "0-7" (contiguous; '
                        "matches how the data was annotated — avoids downloading the whole dataset)")
    p.add_argument("--grades", default=None, help="(blind mode) JSON file to record grades into")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)
    session = build_session(args.data, args.source, args.target, args.grades, args.camera_key,
                            episodes=getattr(args, "episodes", None))
    serve(session, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>robolabel inspect</title><style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2933;--muted:#64748b;--line:#d9dee7;--blue:#2563eb;--red:#b91c1c;--green:#16835b}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
header{height:50px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:5}
header h1{font-size:15px;margin:0}header .meta{color:var(--muted);font-size:13px;display:flex;gap:14px}
main{height:calc(100vh - 50px);display:grid;grid-template-columns:230px minmax(460px,1fr) minmax(360px,440px);gap:10px;padding:10px}
aside,section{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden;min-height:0}
.queue{display:flex;flex-direction:column}.qtools{padding:8px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap}
select,input[type=text]{border:1px solid var(--line);border-radius:6px;padding:5px;font-size:12px}
.qlist{overflow-y:auto;padding:6px}.qi{width:100%;text-align:left;border:1px solid transparent;background:transparent;padding:7px;border-radius:6px;cursor:pointer;color:var(--ink)}
.qi:hover{background:#f1f5f9}.qi.active{border-color:var(--blue);background:#eff6ff}.qid{font-size:12px;font-weight:700}
.qsub{font-size:11px;color:var(--muted);margin-top:2px}.flag{color:var(--red)}
.viewer{display:flex;flex-direction:column}.vtitle{padding:10px 12px 0}.vtitle h2{margin:0 0 2px;font-size:16px}.vtitle p{margin:0;color:var(--muted);font-size:12px}
.stage{padding:10px 12px;display:flex;flex-direction:column;align-items:center}
#frame{width:100%;max-height:46vh;object-fit:contain;background:#0b1220;border-radius:6px}
.scrub{width:100%;margin-top:8px;display:flex;gap:8px;align-items:center}.scrub input[type=range]{flex:1}
.btn{background:#fff;color:var(--ink);border:1px solid var(--line);padding:6px 10px;border-radius:6px;cursor:pointer;font-size:13px}
.btn.play{min-width:60px;font-weight:700}.fcount{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12px;min-width:92px;text-align:right}
.tracks{padding:4px 12px 10px;overflow-x:hidden}
.trk{margin:7px 0}.trkhead{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:3px}
.trkname{font-weight:700;color:var(--ink)}.bar{display:flex;width:100%;height:26px;border:1px solid var(--line);border-radius:5px;overflow:hidden;cursor:pointer}
.seg{height:100%;color:#fff;font-size:10px;padding:2px 3px;overflow:hidden;display:flex;align-items:center;justify-content:center;text-align:center;border-right:1px solid rgba(255,255,255,.35);white-space:nowrap}
.playhead{position:relative}.phline{position:absolute;top:0;bottom:0;width:2px;background:#111827;z-index:3;pointer-events:none}
.right{overflow-y:auto;padding:12px}.tabbar{display:flex;gap:6px;margin-bottom:10px}.tab{flex:1;text-align:center;padding:7px;border:1px solid var(--line);border-radius:6px;cursor:pointer;font-size:13px}
.tab.on{border-color:var(--blue);background:#eff6ff;font-weight:700}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}.num{font-variant-numeric:tabular-nums;text-align:right}
.ev{border:1px solid var(--line);border-radius:7px;padding:8px;margin:8px 0;display:grid;grid-template-columns:120px 1fr;gap:9px;align-items:start}
.ev img{width:120px;border-radius:5px;background:#0b1220;cursor:pointer}.ev .evtxt{font-size:13px}.ev .evf{font-size:11px;color:var(--muted);margin-bottom:3px}
.judge{display:flex;gap:6px;margin-top:6px}.judge button{flex:1;padding:5px;font-size:12px}
.j-yes.sel{background:#dcfce7;border-color:var(--green);color:var(--green);font-weight:700}.j-no.sel{background:#fee2e2;border-color:var(--red);color:var(--red);font-weight:700}
h3{margin:13px 0 6px;font-size:13px}.muted{color:var(--muted);font-size:12px}
.verdict label{display:block;border:1px solid var(--line);border-radius:6px;padding:7px;margin:5px 0;cursor:pointer}
.verdict label:has(input:checked){border-color:var(--blue);background:#eff6ff}
button.primary{background:var(--blue);color:#fff;border:0;padding:10px;border-radius:6px;font-weight:700;cursor:pointer;width:100%;margin-top:8px}
.badge{display:inline-block;background:#eef2ff;color:#3730a3;border-radius:999px;padding:1px 7px;font-size:11px}
</style></head><body>
<header><h1>robolabel · inspect</h1><div class="meta" id="meta"></div></header>
<main>
 <aside class="queue">
  <div class="qtools">
   <select id="sort" onchange="renderQueue()">
    <option value="iou">worst grounded IoU first</option>
    <option value="flags">most gate flags</option>
    <option value="id">episode id</option></select>
   <select id="filter" onchange="renderQueue()">
    <option value="all">all episodes</option>
    <option value="flagged">gate-flagged only</option>
    <option value="ungraded">ungraded only</option></select>
  </div>
  <div class="qlist" id="queue"></div></aside>
 <section class="viewer">
  <div class="vtitle"><h2 id="epid">Loading…</h2><p id="task"></p></div>
  <div class="stage"><img id="frame" alt="frame">
   <div class="scrub"><button class="btn play" id="play" onclick="togglePlay()">▶</button>
    <input type="range" id="slider" min="0" max="0" value="0" oninput="onScrub()">
    <span class="fcount" id="fcount">0 / 0</span></div></div>
  <div class="tracks" id="tracks"></div>
 </section>
 <section class="right">
  <div class="tabbar"><div class="tab on" data-tab="metrics" onclick="tab('metrics')">Metrics</div>
   <div class="tab" data-tab="evidence" onclick="tab('evidence')">Evidence</div>
   <div class="tab" data-tab="grade" id="gradeTab" onclick="tab('grade')" style="display:none">Grade</div></div>
  <div id="panel"></div></section></main>
<script>
let S=null,E=null,cur=null,frame=0,playing=false,timer=null,TAB='metrics';
async function j(u,o={}){const r=await fetch(u,o);const d=await r.json();if(!r.ok||d.error)throw new Error(d.error||r.status);return d;}
async function init(){S=await j('/api/state');document.getElementById('meta').innerHTML=
  `<span>${esc(S.dataset)}</span><span>${S.episodes.length} episodes</span>${S.blind?'<span class="badge">BLIND TRIAL '+S.graded_count+'/'+S.episodes.length+' graded</span>':''}`;
 if(S.blind)document.getElementById('gradeTab').style.display='';
 renderQueue();cur=queueOrder()[0];if(cur)await load(cur);}
function queueOrder(){let q=[...S.episodes];const flt=document.getElementById('filter').value;
 if(flt==='flagged')q=q.filter(e=>e.n_flags>0);if(flt==='ungraded')q=q.filter(e=>!e.graded);
 const s=document.getElementById('sort').value;
 if(s==='iou')q.sort((a,b)=>a.sort_iou-b.sort_iou);else if(s==='flags')q.sort((a,b)=>b.n_flags-a.n_flags);else q.sort((a,b)=>(''+a.episode_id).localeCompare(''+b.episode_id,undefined,{numeric:true}));
 return q.map(e=>e.episode_id);}
function renderQueue(){const order=queueOrder();const root=document.getElementById('queue');root.innerHTML='';
 const byId=Object.fromEntries(S.episodes.map(e=>[e.episode_id,e]));
 for(const id of order){const e=byId[id];const b=document.createElement('button');b.className=`qi ${id===cur?'active':''}`;b.onclick=()=>load(id);
  b.innerHTML=`<div class="qid">${e.graded?'✓ ':''}ep ${esc(id)}</div><div class="qsub">${S.blind?'':'min IoU '+e.sort_iou.toFixed(2)+' · '}${e.n_flags?'<span class="flag">'+e.n_flags+' flags</span>':'no flags'}</div>`;
  root.appendChild(b);}}
async function load(id){stop();cur=id;E=await j('/api/episode/'+encodeURIComponent(id));renderQueue();
 document.getElementById('epid').textContent='Episode '+E.episode_id;document.getElementById('task').textContent=E.task||'';
 const sl=document.getElementById('slider');sl.max=Math.max(0,E.num_frames-1);frame=0;sl.value=0;showFrame();renderTracks();renderPanel();}
function frameEp(){return E.frame_ep||E.episode_id;}
function showFrame(){document.getElementById('frame').src=`/frame/${encodeURIComponent(frameEp())}/${frame}`;
 document.getElementById('fcount').textContent=`${frame} / ${E.num_frames-1}`;updatePlayheads();highlight();}
function onScrub(){frame=Number(document.getElementById('slider').value);showFrame();}
function seekFrame(fr){frame=Math.max(0,Math.min(E.num_frames-1,fr));document.getElementById('slider').value=frame;showFrame();}
function togglePlay(){playing?stop():play();}
function play(){playing=true;document.getElementById('play').textContent='❚❚';const fps=Math.min(E.fps||10,20);
 timer=setInterval(()=>{if(frame>=E.num_frames-1){stop();return;}frame++;document.getElementById('slider').value=frame;showFrame();},1000/fps);}
function stop(){playing=false;if(timer)clearInterval(timer);timer=null;const p=document.getElementById('play');if(p)p.textContent='▶';}
function trackList(){return S.blind?[gName()]:S.track_order;}
function gName(){return S.track_order.find(t=>t!=='gold'&&t!=='S0-Flash'&&t!=='S_grip'&&t!=='uniform-fifths')||S.track_order.find(t=>t!=='gold');}
function renderTracks(){const root=document.getElementById('tracks');root.innerHTML='';const tot=Math.max(1,E.num_frames);
 for(const name of trackList()){const t=E.tracks[name];if(!t)continue;const label=S.blind?'Model (hidden)':name;
  const wrap=document.createElement('div');wrap.className='trk';
  const m=(E.metrics&&E.metrics[name])||{};const mtxt=(name==='gold'||S.blind)?'':`IoU ${fmt(m.iou)} · R@5 ${fmt(m.boundary_recall)}`;
  wrap.innerHTML=`<div class="trkhead"><span class="trkname" style="color:${S.track_colors[name]||'#333'}">${esc(label)}</span><span>${mtxt}</span></div>`;
  const bar=document.createElement('div');bar.className='bar playhead';bar.dataset.name=name;
  for(const s of t.segments){const d=document.createElement('div');d.className='seg';
   d.style.width=Math.max(1.2,((s.end-s.start+1)/tot)*100)+'%';d.style.background=S.track_colors[name]||'#888';
   const lab=segLabel(s);d.textContent=lab;d.title=`${s.start}-${s.end}: ${lab}`+(s.evidence?(' — '+s.evidence):'');
   d.onclick=()=>seekFrame(s.start);bar.appendChild(d);}
  const ph=document.createElement('div');ph.className='phline';bar.appendChild(ph);wrap.appendChild(bar);root.appendChild(wrap);}
 updatePlayheads();}
function updatePlayheads(){if(!E)return;const pct=(frame/Math.max(1,E.num_frames-1))*100;
 for(const ph of document.querySelectorAll('.phline'))ph.style.left=pct+'%';}
function highlight(){/* segment hover handled by title; playhead line shows position */}
function tab(t){TAB=t;for(const el of document.querySelectorAll('.tab'))el.classList.toggle('on',el.dataset.tab===t);renderPanel();}
function renderPanel(){const p=document.getElementById('panel');if(TAB==='metrics')p.innerHTML=metricsHTML();
 else if(TAB==='evidence')renderEvidence(p);else renderGrade(p);}
function metricsHTML(){let q=E.quality||{};let rows='';for(const name of S.track_order){if(name==='gold')continue;const m=(E.metrics||{})[name]||{};
  rows+=`<tr><td style="color:${S.track_colors[name]}">${esc(S.blind?'model':name)}</td><td class="num">${fmt(m.iou)}</td><td class="num">${fmt(m.boundary_precision)}</td><td class="num">${fmt(m.boundary_recall)}</td><td class="num">${m.mae==null?'–':m.mae.toFixed(1)}</td><td class="num">${m.n_segments}</td></tr>`;}
 const flags=(E.gate_flags||[]).map(f=>`<span class="flag">${esc(f)}</span>`).join(' · ')||'<span class="muted">none</span>';
 const ql=`auto ${q.auto??'–'} / gold ${q.gold??'–'}`;const cost=q.cost!=null?('$'+Number(q.cost).toFixed(4)):'–';
 return `<table><tr><th>track</th><th class="num">IoU</th><th class="num">bP±5</th><th class="num">bR±5</th><th class="num">MAE</th><th class="num">#seg</th></tr>${rows}</table>
  <h3>gate flags</h3><div>${flags}</div><h3>quality</h3><div class="muted">${ql} · grounded cost/ep ${cost}</div>
  <h3>gold boundaries</h3><div class="muted">${(E.tracks.gold?boundariesOf(E.tracks.gold):[]).join(', ')||'(single segment / none)'}</div>`;}
function boundariesOf(t){return t.segments.slice(0,-1).map(s=>s.end);}
function renderEvidence(p){const name=gName();const t=E.tracks[name];if(!t){p.innerHTML='<div class="muted">no grounded track</div>';return;}
 p.innerHTML=`<div class="muted">Each evidence string with a thumbnail of its cited frame (segment end). Click the image to jump there and judge whether the claim is true of that frame.</div>`;
 t.segments.forEach((s,i)=>{if(s.evidence==null&&s.phase==null)return;const fr=s.end;const ev=document.createElement('div');ev.className='ev';
  ev.innerHTML=`<img src="/frame/${encodeURIComponent(frameEp())}/${fr}" onclick="seekFrame(${fr})">
   <div class="evtxt"><div class="evf">seg ${i} · <b>${esc(segLabel(s)||'–')}</b> · cited frame ${fr}</div>${esc(s.evidence||'(no evidence string)')}
   ${S.blind?evJudge('ev'+i):''}</div>`;p.appendChild(ev);});}
function evJudge(key){return `<div class="judge"><button class="btn j-yes" onclick="setG('${key}',true,this)">true</button><button class="btn j-no" onclick="setG('${key}',false,this)">false</button></div>`;}
let G={};
function setG(key,val,btn){G[key]=val;const par=btn.parentElement;par.querySelector('.j-yes').classList.toggle('sel',val===true);par.querySelector('.j-no').classList.toggle('sel',val===false);}
function renderGrade(p){const name=gName();const t=E.tracks[name];G=(E.existing_grade&&E.existing_grade.marks)||{};
 let bh='';t.segments.slice(0,-1).forEach((s,i)=>{bh+=`<div class="ev" style="grid-template-columns:96px 1fr"><img src="/frame/${encodeURIComponent(frameEp())}/${s.end}" onclick="seekFrame(${s.end})">
   <div class="evtxt"><div class="evf">boundary ${i} at frame ${s.end} (${esc(segLabel(s)||'–')})</div>
   <div>boundary within ±5 of truth?${judgeBtns('b'+i)}</div><div style="margin-top:5px">phase label correct?${judgeBtns('p'+i)}</div>
   <div style="margin-top:5px">evidence true of frame?${judgeBtns('e'+i)}</div></div></div>`;});
 p.innerHTML=`<div class="muted">Blind: track identity hidden. Grade against the VIDEO, not any reference. Scrub freely.</div>
  <h3>Boundaries & phases & evidence</h3>${bh}
  <h3>Overall verdict</h3><div class="verdict">
   ${vOpt('usable','usable as conditioning data')}${vOpt('touchup','needs touch-up')}${vOpt('garbage','garbage')}</div>
  <button class="primary" onclick="saveGrade()">Save grade & next ungraded</button>`;
 for(const [k,v] of Object.entries(G)){const par=document.querySelector(`[data-k="${k}"]`);if(par){par.querySelector('.j-yes').classList.toggle('sel',v===true);par.querySelector('.j-no').classList.toggle('sel',v===false);}}
 const ver=(E.existing_grade&&E.existing_grade.verdict);if(ver){const r=document.querySelector(`input[name=verdict][value=${ver}]`);if(r)r.checked=true;}}
function judgeBtns(key){return `<span class="judge" data-k="${key}" style="display:inline-flex;width:160px"><button class="btn j-yes" onclick="setG('${key}',true,this)">yes</button><button class="btn j-no" onclick="setG('${key}',false,this)">no</button></span>`;}
function vOpt(v,label){return `<label><input type="radio" name="verdict" value="${v}"> ${label}</label>`;}
async function saveGrade(){const verdict=document.querySelector('input[name=verdict]:checked')?.value||null;
 const payload={episode_id:E.episode_id,track:gName(),marks:G,verdict};
 let res;
 try{res=await j('/api/grade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});}
 catch(err){alert('Grade NOT saved: '+err.message+'\n\nRelaunch the viewer with --grades <file> so grades can be recorded.');return;}
 if(!res.saved){alert('Grade NOT saved: '+(res.error||'unknown')+'\n\nRelaunch with --grades <file>.');return;}
 S=await j('/api/state');document.getElementById('meta').innerHTML=`<span>${esc(S.dataset)}</span><span>${S.episodes.length} episodes</span><span class="badge">BLIND TRIAL ${S.graded_count}/${S.episodes.length} graded · saved to grades file</span>`;
 const ung=S.episodes.find(e=>!e.graded);renderQueue();if(ung)load(ung.episode_id);else alert('All items graded. Now run:\n  robolabel trial-report --grades <file> --unblind fresh_stacking/blind.unblind.json');}
function segLabel(s){return s.phase?(s.phase+(s.target?(' → '+s.target):'')):(s.text||'');}
function fmt(v){return v==null?'–':Number(v).toFixed(2);}
function esc(t){return String(t==null?'':t).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
document.addEventListener('keydown',e=>{if(e.target.matches('input,textarea,select'))return;
 if(e.key==='ArrowRight'){e.preventDefault();seekFrame(frame+1);}if(e.key==='ArrowLeft'){e.preventDefault();seekFrame(frame-1);}
 if(e.key===' '){e.preventDefault();togglePlay();}});
init().catch(err=>{document.body.innerHTML='<pre style="padding:20px;color:#b91c1c">'+esc(err.message)+'</pre>';});
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
