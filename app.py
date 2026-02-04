# ================================
# app.py – Dual Expert Flask App ✅ (Render-ready: auto-download .pth from HF)
# ================================

from flask import Flask, render_template, request, jsonify, send_file
import os, time, ast, traceback, gc, base64, mimetypes, re
from typing import List, Tuple, Optional
from io import BytesIO

import requests

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import XLMRobertaTokenizerFast, XLMRobertaModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.graph_objs as go
import plotly.io as pio

from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException
from openai import OpenAI


# ----------------------------
# CONFIG
# ----------------------------
app = Flask(__name__)

SHOW_DEBUG = False
EXPORT_INTERNAL_COLS = False

APP_NAME = "Catégorisation de produits Auchan"
UI_MODE = "final_only"

# ✅ IMPORTANT: paths compatibles Render (pas de chemins Windows)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACKGROUND_IMAGE = "/static/imageeco.jpg"

GLOBAL_CSV  = os.path.join(BASE_DIR, "data", "produits_nettoyes.csv")
PHASE2_CKPT = os.path.join(BASE_DIR, "models", "domain_expert_flat.pth")

DOMAIN_CSV = os.path.join(BASE_DIR, "data", "domain_expert_phase2_top10_predictions_full (1).csv")
DUEL_CSV   = os.path.join(BASE_DIR, "data", "duel_expert_mistral_GENERAL_EXPERT_top10_corrected (1).csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 160

CONF_THRESHOLD = 0.95
MARGIN_THRESHOLD = 0.12
MIN_TEXT_CHARS = 18
HIGH_CONF_SKIP = 0.98
HIGH_MARGIN_SKIP = 0.20

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8MB

OPENAI_MODEL_VISION = "gpt-4o-mini"
OPENAI_MODEL_TEXT   = "gpt-4o-mini"
OPENAI_TIMEOUT_SEC  = 25
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT_SEC)

BATCH_SIZE_DEFAULT = 16
ENABLE_LLM_IN_BATCH = False


# ----------------------------
# ALWAYS JSON FOR /api/*
# ----------------------------
@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify({
            "error": f"{type(e).__name__}: {e.description}",
            "path": request.path
        }), e.code
    return e

@app.errorhandler(Exception)
def handle_any_exception(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "error": f"{type(e).__name__}: {str(e)}",
            "path": request.path,
            "trace": traceback.format_exc().splitlines()[-15:]
        }), 500
    raise e


# ----------------------------
# UTILS
# ----------------------------
def safe_exists(path: str) -> bool:
    return isinstance(path, str) and path.strip() and os.path.exists(path)

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))

def clamp_int(x, lo, hi, default):
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


# ----------------------------
# HF MODEL DOWNLOAD (Render-friendly)
# ----------------------------
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/Bineta123/domain-expert-xlmr/resolve/main/domain_expert_flat.pth"
).strip()

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()  # seulement si repo HF privé

def download_file(url: str, dst_path: str, timeout_sec: int = 600):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    print(f"[DL] Downloading: {url}")
    with requests.get(url, headers=headers, stream=True, timeout=timeout_sec) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[DL] Saved -> {dst_path}")

def ensure_model_file():
    # Sur Render: le fichier n'est pas dans GitHub → on le télécharge
    if safe_exists(PHASE2_CKPT):
        return
    if not MODEL_URL:
        raise FileNotFoundError("MODEL_URL non défini et checkpoint introuvable.")
    download_file(MODEL_URL, PHASE2_CKPT)


# ----------------------------
# OPTIONAL METRICS + PLOT
# ----------------------------
domain_metrics = None
duel_metrics = None
plot_html = None
df_duel = None

try:
    if safe_exists(DOMAIN_CSV) and safe_exists(DUEL_CSV):
        df_domain = pd.read_csv(DOMAIN_CSV, on_bad_lines="skip")
        df_duel = pd.read_csv(DUEL_CSV, on_bad_lines="skip")

        for df in [df_domain, df_duel]:
            if "description" in df.columns and "text" not in df.columns:
                df.rename(columns={"description": "text"}, inplace=True)
            if "top10_preds" in df.columns:
                df["top10_preds"] = df["top10_preds"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )

        def compute_top1_metrics(df, pred_col):
            y_true = df["true_label"]
            y_pred = df[pred_col]
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)
            }

        def compute_top10_accuracy(df, top10_col):
            return df.apply(lambda row: row["true_label"] in (row[top10_col] or [])[:10], axis=1).mean()

        domain_metrics = compute_top1_metrics(df_domain, "top1_pred")
        domain_metrics["top10_acc"] = compute_top10_accuracy(df_domain, "top10_preds")

        duel_metrics = compute_top1_metrics(df_duel, "final_duel_pred")
        duel_metrics["top10_acc"] = compute_top10_accuracy(df_duel, "top10_preds")

        def create_metrics_bar(d1, d2):
            categories = ["Accuracy Top-1", "Top-10 Accuracy", "F1 Macro"]
            fig = go.Figure(data=[
                go.Bar(name="Domain Expert", x=categories, y=[d1["accuracy"], d1["top10_acc"], d1["f1"]]),
                go.Bar(name="Dual Expert", x=categories, y=[d2["accuracy"], d2["top10_acc"], d2["f1"]]),
            ])
            fig.update_layout(
                title="📊 Comparaison Domain vs Dual Expert",
                yaxis=dict(title="Valeur métrique"),
                barmode="group",
                template="plotly_white",
                height=380
            )
            return pio.to_html(fig, full_html=False)

        plot_html = create_metrics_bar(domain_metrics, duel_metrics)

except Exception as e:
    print("[WARN] Metrics/plot disabled:", type(e).__name__, str(e))
    df_duel = None
    domain_metrics = None
    duel_metrics = None
    plot_html = None


# ----------------------------
# DOMAIN EXPERT MODEL
# ----------------------------
class DomainExpert(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        self.xlm = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt
    for k in ["model_state_dict", "state_dict", "model_state"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    any_tensor = any(torch.is_tensor(v) for v in ckpt.values())
    if any_tensor:
        return ckpt
    raise TypeError("Checkpoint dict reconnu mais aucun state_dict trouvé.")


# ----------------------------
# OpenAI helpers
# ----------------------------
def image_file_to_data_url(filepath: str) -> str:
    mime, _ = mimetypes.guess_type(filepath)
    if not mime:
        mime = "image/jpeg"
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def llm_describe_image_openai(image_data_url: str) -> str:
    if not has_openai_key():
        return ""

    prompt = (
        "Tu es un assistant e-commerce.\n"
        "Décris le produit visible sur l’image en FRANÇAIS, en 1 à 2 phrases courtes.\n"
        "Sois concret : type de produit + attributs clés (couleur, matière, taille si visible).\n"
        "N’invente pas la marque si elle n’est pas clairement lisible.\n"
        "Ne fais pas de liste, seulement une description.\n"
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_VISION,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }],
            max_output_tokens=120
        )
        return (resp.output_text or "").strip()
    except Exception as e:
        print("[OPENAI] describe_image error:", type(e).__name__, str(e))
        return ""

def llm_verify_or_correct_openai(text: str, top1: str, candidates: List[str]) -> Tuple[str, bool]:
    if not candidates or not has_openai_key():
        return top1, False

    prompt = f"""
Tu es un EXPERT GÉNÉRAL en catégorisation e-commerce.

Tâche :
- Vérifie si TOP-1 est correct pour ce produit.
- Si TOP-1 est correct : renvoie TOP-1 exactement.
- Sinon : renvoie la meilleure alternative parmi les candidats.

Description du produit (FR) :
{text}

TOP-1 (prédiction actuelle) :
{top1}

Catégories candidates (choisir UNIQUEMENT dans cette liste) :
{", ".join(candidates)}

Règles STRICTES :
- Renvoie UNE SEULE catégorie exactement comme écrite dans la liste.
- Aucune explication.
- Aucun autre texte.

Catégorie finale :
""".strip()

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        out = (resp.choices[0].message.content or "").strip()
        out_low = out.lower()

        for c in candidates:
            if c.lower() == out_low:
                return c, (c != top1)

        for c in candidates:
            if c.lower() in out_low:
                return c, (c != top1)

        return top1, False
    except Exception as e:
        print("[OPENAI] verify_or_correct error:", type(e).__name__, str(e))
        return top1, False


# ----------------------------
# LOAD (LAZY)
# ----------------------------
tokenizer: Optional[XLMRobertaTokenizerFast] = None
le: Optional[LabelEncoder] = None
model: Optional[DomainExpert] = None
MODEL_READY = False
MODEL_ERROR = None

def load_everything():
    global tokenizer, le, model, MODEL_READY, MODEL_ERROR

    if MODEL_READY:
        return

    try:
        if not safe_exists(GLOBAL_CSV):
            raise FileNotFoundError(f"GLOBAL_CSV introuvable: {GLOBAL_CSV}")

        # ✅ Render: télécharge le .pth si pas présent
        ensure_model_file()
        if not safe_exists(PHASE2_CKPT):
            raise FileNotFoundError(f"Checkpoint introuvable après download: {PHASE2_CKPT}")

        tokenizer_local = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

        df_global = pd.read_csv(GLOBAL_CSV, sep=";", on_bad_lines="skip")
        df_global["taxonomy_path"] = df_global["taxonomy_path"].astype(str)
        le_local = LabelEncoder()
        le_local.fit(df_global["taxonomy_path"].tolist())
        n_classes = len(le_local.classes_)

        model_local = DomainExpert(n_classes).to(DEVICE)
        ckpt = torch.load(PHASE2_CKPT, map_location="cpu")
        sd = extract_state_dict(ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model_local.load_state_dict(sd, strict=False)
        model_local.eval()

        tokenizer = tokenizer_local
        le = le_local
        model = model_local

        del ckpt, sd, df_global
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        MODEL_READY = True
        MODEL_ERROR = None

    except Exception as e:
        MODEL_READY = False
        MODEL_ERROR = f"{type(e).__name__}: {str(e)}"
        raise


# ----------------------------
# PREDICTION HELPERS
# ----------------------------
@torch.no_grad()
def predict_topk_with_probs(text: str, k: int = 10):
    if not MODEL_READY:
        load_everything()
    assert tokenizer is not None and model is not None and le is not None

    enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
    probs = F.softmax(logits, dim=-1).squeeze(0)

    k = max(1, min(int(k), int(probs.shape[0])))
    topk_probs, topk_idx = torch.topk(probs, k=k)

    labels = le.inverse_transform(topk_idx.detach().cpu().numpy()).tolist()
    probs_list = topk_probs.detach().cpu().numpy().tolist()
    return labels, probs_list

@torch.no_grad()
def predict_topk_batch(texts: List[str], k: int = 10, batch_size: int = 16):
    if not MODEL_READY:
        load_everything()
    assert tokenizer is not None and model is not None and le is not None

    all_labels: List[List[str]] = []
    all_probs: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs = F.softmax(logits, dim=-1)

        kk = max(1, min(int(k), int(probs.shape[1])))
        topk_probs, topk_idx = torch.topk(probs, k=kk, dim=1)

        for b in range(topk_idx.size(0)):
            labels = le.inverse_transform(topk_idx[b].detach().cpu().numpy()).tolist()
            probs_list = topk_probs[b].detach().cpu().numpy().tolist()
            all_labels.append(labels)
            all_probs.append(probs_list)

    return all_labels, all_probs


def auto_finalize(text: str, k: int = 10):
    top_labels, top_probs = predict_topk_with_probs(text, k=k)
    top1 = top_labels[0]
    conf1 = float(top_probs[0])
    conf2 = float(top_probs[1]) if len(top_probs) > 1 else 0.0
    margin12 = conf1 - conf2
    too_short = len(text.strip()) < MIN_TEXT_CHARS

    need_llm = False
    if has_openai_key():
        if conf1 >= HIGH_CONF_SKIP and margin12 >= HIGH_MARGIN_SKIP and not too_short:
            need_llm = False
        else:
            if conf1 < CONF_THRESHOLD or margin12 < MARGIN_THRESHOLD or too_short:
                need_llm = True

    final_cat = top1
    used_llm = False

    if need_llm:
        final_cat, _changed = llm_verify_or_correct_openai(text=text, top1=top1, candidates=top_labels[:10])
        used_llm = True

    debug = {
        "confidence_top1": round(conf1, 4),
        "confidence_top2": round(conf2, 4),
        "margin_top1_top2": round(margin12, 4),
        "need_llm": need_llm,
        "llm_called": used_llm,
    }
    return top1, final_cat, top_labels, top_probs, conf1, used_llm, debug


def auto_finalize_force_verify(text: str, k: int = 10):
    top_labels, top_probs = predict_topk_with_probs(text, k=k)
    top1 = top_labels[0]
    conf1 = float(top_probs[0])
    conf2 = float(top_probs[1]) if len(top_probs) > 1 else 0.0
    margin12 = conf1 - conf2

    final_cat = top1
    used_llm = False

    if has_openai_key():
        final_cat, _changed = llm_verify_or_correct_openai(text=text, top1=top1, candidates=top_labels[:10])
        used_llm = True

    debug = {
        "confidence_top1": round(conf1, 4),
        "confidence_top2": round(conf2, 4),
        "margin_top1_top2": round(margin12, 4),
        "forced_verify": True,
        "llm_called": used_llm,
    }
    return top1, final_cat, top_labels, top_probs, conf1, used_llm, debug


# ----------------------------
# Batch export cache
# ----------------------------
LAST_BATCH_DF = None
LAST_BATCH_NAME = "export.csv"


# ----------------------------
# Leaderboard helpers
# ----------------------------
def build_leaderboard():
    rows = []

    if domain_metrics is not None:
        rows.append({
            "model": "DomainExpert (XLM-R large)",
            "accuracy_top1": round(float(domain_metrics["accuracy"]), 4),
            "f1_macro": round(float(domain_metrics["f1"]), 4),
            "top10_acc": round(float(domain_metrics["top10_acc"]), 4),
            "notes": "Local .pth"
        })

    if duel_metrics is not None:
        rows.append({
            "model": "DualExpert (Domain + LLM verify)",
            "accuracy_top1": round(float(duel_metrics["accuracy"]), 4),
            "f1_macro": round(float(duel_metrics["f1"]), 4),
            "top10_acc": round(float(duel_metrics["top10_acc"]), 4),
            "notes": "LLM verify/correct"
        })

    for r in rows:
        r["score"] = round(0.5*r["f1_macro"] + 0.3*r["accuracy_top1"] + 0.2*r["top10_acc"], 4)

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows


def get_example_products(n=10):
    products = []
    if df_duel is not None:
        tmp = df_duel.head(n).copy()
        if "description" not in tmp.columns:
            if "text" in tmp.columns:
                tmp["description"] = tmp["text"].astype(str)
            else:
                tmp["description"] = ""
        products = tmp.to_dict(orient="records")
    return products


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/api/ping", methods=["GET"])
def api_ping():
    return jsonify({
        "ok": True,
        "app_name": APP_NAME,
        "ui_mode": UI_MODE,
        "model_ready": MODEL_READY,
        "model_error": MODEL_ERROR,
        "device": DEVICE,
        "ckpt_exists": safe_exists(PHASE2_CKPT),
        "global_exists": safe_exists(GLOBAL_CSV),
        "has_openai_key": has_openai_key(),
        "conf_threshold": CONF_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "show_debug": bool(SHOW_DEBUG),
        "batch_size_default": BATCH_SIZE_DEFAULT,
        "llm_in_batch": bool(ENABLE_LLM_IN_BATCH)
    })

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html", background_image=BACKGROUND_IMAGE, app_name=APP_NAME)

@app.route("/playground", methods=["GET"])
def playground():
    return render_template(
        "domain_expert.html",
        background_image=BACKGROUND_IMAGE,
        plot_html=None,
        products=None,
        domain_metrics=None,
        duel_metrics=None,
        app_name=APP_NAME
    )

@app.route("/leaderboard", methods=["GET"])
def leaderboard_page():
    return render_template(
        "leaderboard.html",
        background_image=BACKGROUND_IMAGE,
        app_name=APP_NAME,
        plot_html=plot_html,
        products=get_example_products(10)
    )

@app.route("/project", methods=["GET"])
def project_page():
    return render_template("project.html", background_image=BACKGROUND_IMAGE, app_name=APP_NAME)

@app.route("/api/leaderboard", methods=["GET"])
def api_leaderboard():
    return jsonify({"rows": build_leaderboard()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    k = clamp_int(data.get("k", 10), 1, 50, 10)

    if not text:
        return jsonify({"error": "Veuillez entrer une description produit."}), 400

    t0 = time.time()
    top1, final_cat, top_labels, top_probs, conf_top1, used_llm, debug = auto_finalize(text=text, k=k)

    payload = {
        "input_text": text,
        "final_category": final_cat,
        "latency_sec": round(time.time() - t0, 2),
        "used_llm": bool(used_llm)
    }

    if SHOW_DEBUG:
        payload.update({
            "proposed_category": top1,
            "debug": debug,
            "top10_labels": top_labels[:10],
            "top10_probs": [round(float(x), 6) for x in top_probs[:10]]
        })

    return jsonify(payload)


@app.route("/api/predict_image", methods=["POST"])
def api_predict_image():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier image reçu (champ 'file')."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Nom de fichier vide."}), 400

    k = clamp_int(request.form.get("k", 10), 1, 50, 10)

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{int(time.time())}_{filename}")
    file.save(save_path)

    t0 = time.time()

    if not has_openai_key():
        return jsonify({
            "error": "OPENAI_API_KEY manquante. Impossible de générer la description image.",
            "image_filename": os.path.basename(save_path)
        }), 400

    image_data_url = image_file_to_data_url(save_path)
    desc = llm_describe_image_openai(image_data_url)

    if not desc:
        return jsonify({
            "error": "Impossible de générer la description (erreur OpenAI).",
            "image_filename": os.path.basename(save_path)
        }), 500

    top1, final_cat, top_labels, top_probs, conf_top1, used_llm, debug = auto_finalize_force_verify(text=desc, k=k)

    payload = {
        "image_filename": os.path.basename(save_path),
        "generated_description": desc,
        "final_category": final_cat,
        "latency_sec": round(time.time() - t0, 2),
        "used_llm": bool(used_llm)
    }

    if SHOW_DEBUG:
        payload.update({
            "proposed_category": top1,
            "debug": debug,
            "top10_labels": top_labels[:10],
            "top10_probs": [round(float(x), 6) for x in top_probs[:10]]
        })

    return jsonify(payload)


@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    global LAST_BATCH_DF, LAST_BATCH_NAME

    if "csv_file" not in request.files:
        return jsonify({"error": "Aucun fichier CSV reçu (champ 'csv_file')."}), 400

    file = request.files["csv_file"]
    if not file.filename:
        return jsonify({"error": "Nom de fichier vide."}), 400

    sep = request.form.get("sep", ";")
    k = clamp_int(request.form.get("k", 10), 1, 50, 10)
    batch_size = clamp_int(request.form.get("batch_size", BATCH_SIZE_DEFAULT), 1, 128, BATCH_SIZE_DEFAULT)

    try:
        df = pd.read_csv(file, sep=sep, on_bad_lines="skip")
    except Exception:
        try:
            file.stream.seek(0)
            df = pd.read_csv(file, sep=",", on_bad_lines="skip")
            sep = ","
        except Exception as e:
            return jsonify({"error": f"Impossible de lire le CSV: {type(e).__name__}: {str(e)}"}), 400

    text_col = None
    for c in ["text", "description"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        return jsonify({"error": "Le CSV doit contenir une colonne 'text' ou 'description'."}), 400

    df[text_col] = df[text_col].astype(str).fillna("").str.strip()
    df = df[df[text_col].str.len() > 0].copy()
    if len(df) == 0:
        return jsonify({"error": "Aucune ligne valide (text/description vide)."}), 400

    texts = df[text_col].tolist()
    t0 = time.time()

    top_labels_all, top_probs_all = predict_topk_batch(texts, k=k, batch_size=batch_size)

    final_list: List[str] = []
    used_llm_list: List[bool] = []

    if ENABLE_LLM_IN_BATCH and has_openai_key():
        for txt in texts:
            _top1, final_cat, *_rest = auto_finalize(text=txt, k=k)
            final_list.append(final_cat)
            used_llm_list.append(True)
    else:
        for top_labels in top_labels_all:
            final_list.append(top_labels[0])
            used_llm_list.append(False)

    df_out = df.copy()
    df_out["final_category"] = final_list
    df_out["used_llm"] = used_llm_list

    if EXPORT_INTERNAL_COLS:
        df_out["top10_alternatives"] = [x[:10] for x in top_labels_all]
        df_out["top10_probs"] = [x[:10] for x in top_probs_all]

    LAST_BATCH_DF = df_out
    LAST_BATCH_NAME = f"export_{int(time.time())}.csv"

    return jsonify({
        "message": "Batch terminé",
        "n": int(len(df_out)),
        "latency_sec": round(time.time() - t0, 2),
        "export_url": "/api/export.csv",
        "batch_size": batch_size,
        "llm_in_batch": bool(ENABLE_LLM_IN_BATCH),
        "ui_mode": UI_MODE
    })

@app.route("/api/export.csv", methods=["GET"])
def api_export_csv():
    global LAST_BATCH_DF, LAST_BATCH_NAME

    if LAST_BATCH_DF is None or len(LAST_BATCH_DF) == 0:
        return jsonify({"error": "Aucun résultat batch disponible. Lance d'abord un batch."}), 400

    buf = BytesIO()
    csv_bytes = LAST_BATCH_DF.to_csv(index=False).encode("utf-8")
    buf.write(csv_bytes)
    buf.seek(0)

    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=LAST_BATCH_NAME)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
