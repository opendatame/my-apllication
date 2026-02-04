import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import XLMRobertaTokenizerFast, XLMRobertaModel

# ----------------------------
# PATHS (Windows)
# ----------------------------
GLOBAL_CSV  = r"C:\Users\DELL\Downloads\Test\data\produits_nettoyes.csv"
PHASE2_CKPT = r"C:\Users\DELL\Downloads\Test\models\domain_expert_flat.pth"

# ----------------------------
# DEVICE
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 160

# ----------------------------
# DOMAIN EXPERT MODEL
# ----------------------------
class DomainExpert(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()
        print("    [3.1] loading backbone: xlm-roberta-large ...")

        # ✅ IMPORTANT: réduit le pic mémoire au chargement
        self.xlm = XLMRobertaModel.from_pretrained(
            "xlm-roberta-large",
            low_cpu_mem_usage=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.xlm.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.xlm(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

def extract_state_dict(ckpt):
    """
    Accepte plusieurs formats de checkpoint:
    - state_dict direct
    - dict avec model_state_dict / state_dict / model_state
    """
    if not isinstance(ckpt, dict):
        return ckpt

    for k in ["model_state_dict", "state_dict", "model_state"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]

    any_tensor = any(torch.is_tensor(v) for v in ckpt.values())
    if any_tensor:
        return ckpt

    raise TypeError("Checkpoint dict reconnu mais aucun state_dict trouvé (model_state_dict/state_dict/model_state).")

def main():
    # ----------------------------
    # CHECK FILES
    # ----------------------------
    if not os.path.exists(GLOBAL_CSV):
        raise FileNotFoundError(f"GLOBAL_CSV introuvable: {GLOBAL_CSV}")
    if not os.path.exists(PHASE2_CKPT):
        raise FileNotFoundError(f"Checkpoint introuvable: {PHASE2_CKPT}")

    # ----------------------------
    # TOKENIZER & LABEL ENCODER
    # ----------------------------
    print("[1] Load tokenizer...")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        "xlm-roberta-large",
        use_fast=True
    )
    print("    ✅ tokenizer OK")

    print("[2] Load LabelEncoder...")
    df_global = pd.read_csv(GLOBAL_CSV, sep=";", on_bad_lines="skip")
    df_global["taxonomy_path"] = df_global["taxonomy_path"].astype(str)

    le = LabelEncoder()
    le.fit(df_global["taxonomy_path"].tolist())
    NUM_CLASSES = len(le.classes_)
    print("    ✅ NUM_CLASSES =", NUM_CLASSES)

    # ----------------------------
    # BUILD MODEL
    # ----------------------------
    print("[3] Build model...")
    model = DomainExpert(NUM_CLASSES).to(DEVICE)
    print("    ✅ model created")

    # ----------------------------
    # LOAD CHECKPOINT
    # ----------------------------
    print("[4] Load checkpoint...")
    ckpt = torch.load(PHASE2_CKPT, map_location="cpu")  # ✅ cpu-safe
    sd = extract_state_dict(ckpt)

    if not isinstance(sd, dict):
        raise TypeError(f"State dict invalide. Type reçu: {type(sd)}")

    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)

    # cleanup
    del ckpt, sd
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    print("[5] ✅ Model loaded and ready.")
    if missing:
        print("    [WARN] Missing keys:", missing[:10], ("..." if len(missing) > 10 else ""))
    if unexpected:
        print("    [WARN] Unexpected keys:", unexpected[:10], ("..." if len(unexpected) > 10 else ""))

    # ----------------------------
    # TEST ONE PREDICTION
    # ----------------------------
    print("[6] Test prediction...")
    text = "iphone 13 128gb blue unlocked charger included"
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

    with torch.no_grad():
        logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE))
        probs = F.softmax(logits, dim=-1).squeeze(0)

    topk = 10
    topk_probs, topk_idx = torch.topk(probs, k=min(topk, int(probs.shape[0])))
    labels = le.inverse_transform(topk_idx.detach().cpu().numpy()).tolist()

    print("\n===== RESULT =====")
    print("TEXT:", text)
    print("TOP1:", labels[0], "| prob =", float(topk_probs[0].item()))
    print("TOP10:")
    for i, (lab, pr) in enumerate(zip(labels, topk_probs.tolist()), 1):
        print(f"  {i:02d}. {lab} (p={pr:.6f})")

if __name__ == "__main__":
    main()
