# app.py
import os, pathlib, functools, time, logging, hashlib, re, csv, io, datetime
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from litellm import completion

# ---- optional XLSX export ----
try:
    from openpyxl import Workbook  # type: ignore
    HAS_XLSX = True
except Exception:
    HAS_XLSX = False

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("sentiment.log")]
)
logger = logging.getLogger("sentiment-mini")

# ---------------- Env / Paths ----------------
ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 20:
    logger.warning("OPENAI_API_KEY seems missing/short. LLM calls may fail.")

# ---------------- Model & Prompts ----------------
MODEL = "gpt-4o-mini"

# İzinli etiketler (TR)
ALLOWED = {"olumlu", "olumsuz-urun", "olumsuz-temizlik", "olumsuz-hizmet", "notr"}

SYSTEM = """You are a strict sentiment classifier for Turkish customer reviews.
    Your task is to classify each review into exactly ONE category.
    Allowed output tokens (choose only one):
    0 (olumlu) | 1 (olumsuz-urun) | 2 (olumsuz-temizlik) | 3 (olumsuz-hizmet) | 4 (notr).
    Rules: Output must be ONLY the chosen token (0, 1, 2, 3, or 4), with no punctuation,
    no extra words, and no explanations.
"""

# Few-shot ile hedefi keskinleştiriyoruz
FEW_SHOTS = [
    {"role": "user", "content": "Yorum: \"ürün kutusu ezilmiş, iade edeceğim\""},
    {"role": "assistant", "content": "olumsuz-urun"},
    {"role": "user", "content": "Yorum: \"personel çok ilgisizdi, sorunumu çözmediler\""},
    {"role": "assistant", "content": "olumsuz-hizmet"},
    {"role": "user", "content": "Yorum: \"oda temizlenmemişti, banyo kirliydi\""},
    {"role": "assistant", "content": "olumsuz-temizlik"},
    {"role": "user", "content": "Yorum: \"kargoda biraz bekledi ama ürün güzel\""},
    {"role": "assistant", "content": "olumlu"},
    {"role": "user", "content": "Yorum: \"bilmem, kararsızım\""},
    {"role": "assistant", "content": "notr"},
]

USER_TMPL = (
    "Aşağıdaki yorumu sınıflandır. Sadece bir etiket döndür:\n"
    "olumlu, olumsuz-urun, olumsuz-temizlik, olumsuz-hizmet veya notr.\n\n"
    "Kurallar:\n"
    "- Ürün/kargo/kalite/ambalaj hatası ➜ olumsuz-urun\n"
    "- Temizlik/hijyen/koku/kir/pislik ➜ olumsuz-temizlik\n"
    "- Personel/servis/destek/ilgisizlik ➜ olumsuz-hizmet\n"
    "- Övgü/teşekkür/memnuniyet ➜ olumlu\n"
    "- Belirsiz/nötr ➜ notr\n"
    "- Karışık ifadelerde NEGATİF etiketler olumluya üstün gelir.\n\n"
    "Yorum:\n\"\"\"{text}\"\"\""
)

# ---------------- FastAPI ----------------
app = FastAPI(title="Sentiment Mini")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    if request.url.scheme == "https":
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "object-src 'none'"
    )
    resp.headers["Content-Security-Policy"] = csp
    return resp

# ---------------- Rate limit ----------------
WINDOW_SEC = 60
MAX_REQ_IN_WINDOW = 30
_hits = defaultdict(deque)

def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > WINDOW_SEC:
        q.popleft()
    if len(q) >= MAX_REQ_IN_WINDOW:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    q.append(now)

# ---------------- Pydantic ----------------
class Item(BaseModel):
    text: str = Field(..., min_length=1, max_length=300)

# ---------------- Utils ----------------
_space_re = re.compile(r"\s+")
def normalize(s: str) -> str:
    return _space_re.sub(" ", s).strip().lower()

def sha256_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --------- Heuristic fallback (LLM “notr” derse devreye girer) ----------
NEG_PRODUCT = {"kargo", "kargoda", "kırık", "bozuk", "iade", "ambalaj", "paket", "gecikti", "gecikme", "uyumadı", "uyumsuz", "çalışmıyor", "arızalı"}
NEG_CLEAN   = {"kirli", "pis", "temizlenmemiş", "hijyen", "kokuyor", "kokuyordu", "leke", "toz", "çöp"}
NEG_SERVICE = {"personel", "ilgisiz", "saygısız", "destek", "musteri hizmet", "müşteri hizmet", "servis", "yardım etmedi", "beklettiler"}
POSITIVE    = {"harika", "mükemmel", "çok iyi", "süper", "beğendim", "memnunum", "teşekkürler", "güzel"}

def heuristic_label(text: str) -> str:
    t = normalize(text)
    neg = False
    # negatif alt sınıf öncelikleri
    if any(w in t for w in NEG_CLEAN):
        return "olumsuz-temizlik"
    if any(w in t for w in NEG_SERVICE):
        return "olumsuz-hizmet"
    if any(w in t for w in NEG_PRODUCT):
        return "olumsuz-urun"
    # pozitif
    if any(w in t for w in POSITIVE):
        return "olumlu"
    return "notr"

# ---------------- LLM (cached) ----------------
@functools.cache
def _llm_only(text: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing (.env içine koy)")

    # Few-shot + talimat
    messages = [{"role": "system", "content": SYSTEM}] + FEW_SHOTS + [
        {"role": "user", "content": USER_TMPL.format(text=text)}
    ]

    resp = completion(
        model=MODEL,
        messages=messages,
        temperature=0,          # deterministik
        max_tokens=3,
        timeout=30,
    )
    raw = (resp.choices[0].message["content"] or "").strip().lower()
    if raw not in ALLOWED:
        # Bozuk çıktı gelirse heuristik
        raw = heuristic_label(text)

    # Eğer LLM "notr" dediyse ama heuristik güçlü sinyal veriyorsa override et
    if raw == "notr":
        hint = heuristic_label(text)
        if hint != "notr":
            raw = hint

    cost = float(getattr(resp, "_hidden_params", {}).get("response_cost", 0.0))
    usage = getattr(resp, "usage", {}) or {}
    return {"label": raw, "cost_usd": round(cost, 6), "usage": usage}

# ---------------- Classify (DB yok) ----------------
def classify_no_db(text_input: str) -> dict:
    text_norm = normalize(text_input)
    res = _llm_only(text_norm)
    return {
        "label": res["label"],
        "usage": res["usage"],
        "cost_usd": res["cost_usd"],
        "model": MODEL,
        "source": "llm",
    }

# ---------------- CSV Helpers & State ----------------
LAST_CSV_RESULT: Optional[Dict[str, Any]] = None  # tekli gösterim için
CSV_STATE = {"offset": 0, "mtime": None, "path": None}
LAST_CSV_BATCH: List[Dict[str, Any]] = []

def _sniff_dict_reader(f) -> csv.DictReader:
    sample = f.read(4096)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
    except Exception:
        dialect = csv.excel
    return csv.DictReader(f, dialect=dialect)

def _read_first_row_with_text(csv_path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not csv_path.exists():
        return None

    encodings = ["utf-8", "utf-8-sig", "iso-8859-9", "windows-1254", "latin-1"]
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, errors="strict") as f:
                reader = _sniff_dict_reader(f)
                field_map = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
                id_key = next((field_map[c] for c in ["review_id", "id"] if c in field_map), None)
                text_key = next((field_map[c] for c in ["before_text_ai", "text", "review_text", "comment"] if c in field_map), None)

                for row in reader:
                    rid = (row.get(id_key) if id_key else "") or "N/A"
                    txt = (row.get(text_key) if text_key else "") or ""
                    txt = str(txt).strip()
                    if txt:
                        return {"review_id": rid, "before_text_ai": txt}
            break
        except Exception as e:
            logger.warning(f"CSV parse failed with {enc}: {e}")
            continue
    return None

def _read_csv_rows(csv_path: pathlib.Path, start_offset: int, limit: int) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    encodings = ["utf-8", "utf-8-sig", "iso-8859-9", "windows-1254", "latin-1"]
    rows: List[Dict[str, Any]] = []

    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc, errors="strict") as f:
                reader = _sniff_dict_reader(f)

                field_map = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
                id_key = next((field_map[c] for c in ["review_id", "id"] if c in field_map), None)
                text_key = next((field_map[c] for c in ["before_text_ai", "text", "review_text", "comment"] if c in field_map), None)

                if not text_key:
                    return rows

                # offset kadar satır atla
                skipped = 0
                for row in reader:
                    if skipped < start_offset:
                        skipped += 1
                        continue
                    rid = (row.get(id_key) if id_key else "") or "N/A"
                    txt = (row.get(text_key) if text_key else "") or ""
                    txt = str(txt).strip()
                    if txt:
                        rows.append({"review_id": rid, "before_text_ai": txt})
                        if len(rows) >= limit:
                            break
            break
        except Exception as e:
            logger.warning(f"CSV parse failed with {enc}: {e}")
            continue

    return rows

# ---------------- Health ----------------
@app.get("/health")
def health_check(_=Depends(rate_limit)):
    return {"status": "ok", "checks": {"openai_api_key": bool(OPENAI_API_KEY)}}

# ---------------- HTML (UI aynı) ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html><meta charset="utf-8">
<title>Sentiment Mini</title>
<style>
  body{font-family:system-ui;margin:24px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  input{width:360px;padding:8px}button{padding:8px 12px; cursor:pointer}
  #meta,#csvMeta{color:#555;white-space:pre-wrap}
  #csvBox, #dlBox { display:none; }
  table{border-collapse:collapse; width:100%; margin-top:8px}
  th, td{border:1px solid #ccc; padding:6px; text-align:left; vertical-align:top}
</style>

<h3>Sentiment Mini</h3>

<div class="row">
  <input id="t" placeholder="Yorum yaz..." />
  <button onclick="go()">Analiz Et</button>
  <button onclick="goCsv()">CSV’den Analiz Et</button>
  <button onclick="goCsvBatch(5)">CSV’den 5 Analiz Et</button>
</div>

<p>Sonuç: <b id="out">-</b></p>
<pre id="meta"></pre>

<hr/>

<h4>CSV Sonuç</h4>
<div id="csvBox">
  <div id="csvText"></div>
  <div id="csvMeta"></div>

  <table id="csvTable" style="display:none">
    <thead>
      <tr>
        <th>Review ID</th>
        <th>Before Text AI</th>
        <th>Sonuç</th>
        <th>Pricing (USD)</th>
        <th>Model</th>
        <th>Source</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <div id="dlBox" style="margin-top:8px">
    <a id="dlLink" href="/export-last-xlsx">Sonucu indir (Excel/CSV)</a>
  </div>
</div>

<script>
async function go(){
  const text = document.getElementById('t').value.trim();
  if(!text){alert('Yorum boş');return;}
  const r = await fetch('/sentiment', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text})
  });
  const j = await r.json();
  if(r.ok){
    document.getElementById('out').textContent = j.label + " ("+ j.source +")";
    document.getElementById('meta').textContent =
      `Model: ${j.model}\\n` +
      `Prompt tokens: ${j.usage?.prompt_tokens ?? ''}\\n` +
      `Completion tokens: ${j.usage?.completion_tokens ?? ''}\\n` +
      `Total tokens: ${j.usage?.total_tokens ?? ''}\\n` +
      `Cost (USD): $${j.cost_usd}`;
  }else{
    document.getElementById('out').textContent = j.detail || 'error';
    document.getElementById('meta').textContent = '';
  }
}

async function goCsv(){
  const r = await fetch('/sentiment-from-csv');
  const j = await r.json();

  const box = document.getElementById('csvBox');
  const meta = document.getElementById('csvMeta');
  const textDiv = document.getElementById('csvText');
  const dlBox = document.getElementById('dlBox');
  const tbl = document.getElementById('csvTable');
  const tbody = tbl.querySelector('tbody');

  box.style.display = 'block';
  tbody.innerHTML = '';
  tbl.style.display = 'none';

  if(r.ok && j.review_id){
    textDiv.textContent =
`Review ID: ${j.review_id}
Before Text AI: ${j.before_text_ai}
Sonuç: ${j.label}
Pricing: $${j.cost_usd}`;
    meta.textContent = '';
    dlBox.style.display = 'block';
  }else{
    textDiv.textContent = j.detail || 'CSV okuma hatası / metin bulunamadı';
    meta.textContent = '';
    dlBox.style.display = 'none';
  }
}

async function goCsvBatch(n){
  const r = await fetch(`/sentiment-from-csv-batch?n=${encodeURIComponent(n)}`);
  const j = await r.json();

  const box = document.getElementById('csvBox');
  const meta = document.getElementById('csvMeta');
  const textDiv = document.getElementById('csvText');
  const dlBox = document.getElementById('dlBox');
  const tbl = document.getElementById('csvTable');
  const tbody = tbl.querySelector('tbody');

  box.style.display = 'block';
  tbody.innerHTML = '';

  if (r.ok && Array.isArray(j.items) && j.items.length){
    textDiv.textContent = `Batch sonuç (${j.items.length} kayıt)`;
    meta.textContent = '';
    j.items.forEach(it => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${it.review_id ?? 'N/A'}</td>
        <td>${(it.before_text_ai ?? '').replaceAll('<','&lt;')}</td>
        <td>${it.label ?? ''}</td>
        <td>${it.cost_usd ?? 0}</td>
        <td>${it.model ?? ''}</td>
        <td>${it.source ?? ''}</td>
      `;
      tbody.appendChild(tr);
    });
    tbl.style.display = 'table';
    dlBox.style.display = 'block';
  } else {
    textDiv.textContent = j.detail || j.message || 'Batch analizi başarısız/boş';
    meta.textContent = '';
    tbl.style.display = 'none';
    dlBox.style.display = 'none';
  }
}
</script>
"""

# ---------------- Core endpoints ----------------
@app.post("/sentiment")
def sentiment(item: Item, _=Depends(rate_limit)):
    try:
        res = classify_no_db(item.text.strip())
        return res
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/sentiment-from-csv")
def sentiment_from_csv(_=Depends(rate_limit)):
    csv_path = ROOT / "sentences.csv"
    row = _read_first_row_with_text(csv_path)
    if not row:
        raise HTTPException(status_code=404, detail="CSV'de uygun metin bulunamadı")

    res = classify_no_db(row["before_text_ai"])
    out = {
        "review_id": row["review_id"],
        "before_text_ai": row["before_text_ai"],
        "label": res["label"],
        "cost_usd": res["cost_usd"],
        "model": res["model"],
        "source": res["source"],
    }
    # son tekli sonuç
    global LAST_CSV_RESULT
    LAST_CSV_RESULT = out
    # tekli indirme için batch'i sıfırla ki yanlışlık olmasın
    LAST_CSV_BATCH.clear()
    return out

@app.get("/sentiment-from-csv-batch")
def sentiment_from_csv_batch(n: int = 5, _=Depends(rate_limit)):
    csv_path = ROOT / "sentences.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="sentences.csv bulunamadı")

    mtime = csv_path.stat().st_mtime
    if CSV_STATE["path"] != str(csv_path) or CSV_STATE["mtime"] != mtime:
        CSV_STATE["path"] = str(csv_path)
        CSV_STATE["mtime"] = mtime
        CSV_STATE["offset"] = 0

    start = CSV_STATE["offset"]
    picked_rows = _read_csv_rows(csv_path, start_offset=start, limit=max(1, n))
    if not picked_rows:
        return {"items": [], "count": 0, "offset": start, "message": "CSV bitti ya da uygun satır yok."}

    results: List[Dict[str, Any]] = []
    for r in picked_rows:
        res = classify_no_db(r["before_text_ai"])
        results.append({
            "review_id": r["review_id"],
            "before_text_ai": r["before_text_ai"],
            "label": res["label"],
            "cost_usd": res["cost_usd"],
            "model": res["model"],
            "source": res["source"],
        })

    CSV_STATE["offset"] = start + len(picked_rows)
    LAST_CSV_BATCH.clear()
    LAST_CSV_BATCH.extend(results)

    return {"items": results, "count": len(results), "offset": CSV_STATE["offset"], "status": "ok"}

@app.get("/export-last-xlsx")
def export_last_xlsx(_=Depends(rate_limit)):
    headers = ["Review ID", "Before Text AI", "Sonuç", "Pricing (USD)", "Model", "Source", "Generated At"]

    rows: List[List[Any]] = []
    if LAST_CSV_BATCH:
        for it in LAST_CSV_BATCH:
            rows.append([
                it.get("review_id", "N/A"),
                it.get("before_text_ai", ""),
                it.get("label", ""),
                it.get("cost_usd", 0.0),
                it.get("model", ""),
                it.get("source", ""),
                datetime.datetime.now().isoformat(timespec="seconds"),
            ])
    elif LAST_CSV_RESULT:
        rows.append([
            LAST_CSV_RESULT.get("review_id", "N/A"),
            LAST_CSV_RESULT.get("before_text_ai", ""),
            LAST_CSV_RESULT.get("label", ""),
            LAST_CSV_RESULT.get("cost_usd", 0.0),
            LAST_CSV_RESULT.get("model", ""),
            LAST_CSV_RESULT.get("source", ""),
            datetime.datetime.now().isoformat(timespec="seconds"),
        ])
    else:
        raise HTTPException(status_code=400, detail="Önce CSV’den analiz yapın (tekli veya batch).")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if HAS_XLSX:
        filename = f"csv_results_{ts}.xlsx"
        out_path = ROOT / filename
        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(headers)
        for r in rows:
            ws.append(r)
        wb.save(out_path)
        return FileResponse(
            out_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=filename
        )
    else:
        filename = f"csv_results_{ts}.csv"
        out_path = ROOT / filename
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return FileResponse(out_path, media_type="text/csv", filename=filename)

# ---------------- Download last single result as CSV (optional) ----------------
_last_csv_payload: Optional[Tuple[str, str, str, float]] = None

@app.get("/download-csv-result")
def download_csv_result():
    """
    (Opsiyonel) Tekli sonucu indir — HTML'deki eski linkle uyumlu.
    """
    global LAST_CSV_RESULT
    if not LAST_CSV_RESULT:
        raise HTTPException(status_code=400, detail="Önce CSV'den analiz yapın.")
    review_id = LAST_CSV_RESULT["review_id"]
    before_text = LAST_CSV_RESULT["before_text_ai"]
    result_tr = LAST_CSV_RESULT["label"]
    price = LAST_CSV_RESULT["cost_usd"]

    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["review_id", "before_text_ai", "sonuc", "pricing_usd"])
    wr.writerow([review_id, before_text, result_tr, f"{price:.6f}"])
    data = buf.getvalue()

    return PlainTextResponse(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="result.csv"'}
    )