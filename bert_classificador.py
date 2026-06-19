"""
BERTimbau ajustado como classificador multirrótulo de alérgenos.

Hiperparâmetros por variável de ambiente: EPOCHS, BATCH, MAXLEN, LR, SUBSET, SEED.

Uso:
    python3 bert_classificador.py
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from conhecimento_alergenos import ALERGENOS
from avaliacao_alergenos import (
    relatorio, split, micro_f1_pares, macro_f1_pares,
    limiares_por_rotulo, preditos_por_rotulo,
)

MODELO = os.environ.get("MODELO", "neuralmind/bert-base-portuguese-cased")
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH = int(os.environ.get("BATCH", 8))
MAXLEN = int(os.environ.get("MAXLEN", 160))
LR = float(os.environ.get("LR", 2e-5))
SUBSET = int(os.environ.get("SUBSET", 0))
SEED = int(os.environ.get("SEED", 42))
SAIDA = "modelo_bert_alergenos"

USA_LORA = os.environ.get("LORA", "1" if "large" in MODELO else "0") == "1"

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construir_modelo():
    """Base congelado quando LoRA."""
    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO, num_labels=len(ALERGENOS),
        problem_type="multi_label_classification")
    if USA_LORA:
        from peft import LoraConfig, get_peft_model, TaskType
        config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32,
                            lora_dropout=0.1, target_modules=["query", "value"],
                            modules_to_save=["classifier"])
        modelo = get_peft_model(modelo, config)
    return modelo.to(DISPOSITIVO)


def semear(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def texto_para_bert(produto):
    """Remove underscores do OFF (marcação de alérgeno que vazaria o rótulo)."""
    return produto["cru"].replace("_", " ")


def tensores(produtos, tokenizer):
    enc = tokenizer([texto_para_bert(p) for p in produtos],
                    truncation=True, padding=True, max_length=MAXLEN,
                    return_tensors="pt")
    y = torch.tensor([p["y"] for p in produtos], dtype=torch.float)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], y)


def pesos_classe(treino):
    """(negativos / positivos) por rótulo, recortado em [1, 20]."""
    Y = np.array([p["y"] for p in treino])
    positivos = Y.sum(axis=0)
    negativos = len(Y) - positivos
    w = negativos / np.clip(positivos, 1, None)
    return torch.tensor(np.clip(w, 1.0, 20.0), dtype=torch.float)


@torch.no_grad()
def probabilidades(model, ds):
    model.eval()
    saidas = []
    for input_ids, mask, _ in DataLoader(ds, batch_size=32, num_workers=0):
        logits = model(input_ids=input_ids.to(DISPOSITIVO),
                       attention_mask=mask.to(DISPOSITIVO)).logits
        saidas.append(torch.sigmoid(logits).cpu())
    return torch.cat(saidas).numpy()


def treinar(model, treino_ds, val_ds, conjunto_val, pesos):
    loader = DataLoader(treino_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    otim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterio = torch.nn.BCEWithLogitsLoss(pos_weight=pesos.to(DISPOSITIVO))
    golds_val = [p["gold"] for p in conjunto_val]

    def estado_cpu():
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    melhor_estado, melhor_f1 = estado_cpu(), -1.0

    for epoca in range(EPOCHS):
        model.train()
        perda_total = 0.0
        for input_ids, mask, y in loader:
            otim.zero_grad()
            logits = model(input_ids=input_ids.to(DISPOSITIVO),
                           attention_mask=mask.to(DISPOSITIVO)).logits
            perda = criterio(logits, y.to(DISPOSITIVO))
            perda.backward()
            otim.step()
            perda_total += perda.item()

        probs_val = probabilidades(model, val_ds)
        limiares = limiares_por_rotulo(probs_val, conjunto_val)
        f1 = micro_f1_pares(list(zip(preditos_por_rotulo(probs_val, limiares), golds_val)))
        print(f"  época {epoca + 1}/{EPOCHS}, perda {perda_total / len(loader):.4f} "
              f", val micro-F1 {f1:.3f}", flush=True)
        if f1 > melhor_f1:
            melhor_f1, melhor_estado = f1, estado_cpu()

    model.load_state_dict(melhor_estado)


def treinar_e_avaliar(seed, treino, validacao, teste, tokenizer):
    semear(seed)
    model = construir_modelo()
    treinar(model, tensores(treino, tokenizer), tensores(validacao, tokenizer),
            validacao, pesos_classe(treino))
    limiares = limiares_por_rotulo(
        probabilidades(model, tensores(validacao, tokenizer)), validacao)
    probs_teste = probabilidades(model, tensores(teste, tokenizer))
    pares = list(zip(preditos_por_rotulo(probs_teste, limiares),
                     [p["gold"] for p in teste]))
    return model, limiares, probs_teste, pares


def main():
    treino, validacao, teste = split()
    if SUBSET:
        treino = treino[:SUBSET]
    sementes = [int(s) for s in os.environ.get("SEEDS", "").split(",") if s] or [SEED]
    print(f"treino={len(treino)} validação={len(validacao)} teste={len(teste)} | "
          f"modelo={MODELO} | lora={USA_LORA} | dispositivo={DISPOSITIVO} | "
          f"epochs={EPOCHS} batch={BATCH} maxlen={MAXLEN} lr={LR} sementes={sementes}",
          flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    micros, macros, pares_primeiro = [], [], None
    for i, seed in enumerate(sementes):
        print(f"treinando (semente {seed})...", flush=True)
        model, limiares, probs_teste, pares = treinar_e_avaliar(
            seed, treino, validacao, teste, tokenizer)
        micros.append(micro_f1_pares(pares))
        macros.append(macro_f1_pares(pares))
        print(f"  semente {seed}: micro-F1 {micros[-1]:.3f} macro-F1 {macros[-1]:.3f}",
              flush=True)
        if i == 0:
            a_salvar = model.merge_and_unload() if USA_LORA else model
            a_salvar.save_pretrained(SAIDA)
            tokenizer.save_pretrained(SAIDA)
            np.save(f"{SAIDA}/probs_teste.npy", probs_teste)
            np.save(f"{SAIDA}/limiares.npy", limiares)
            pares_primeiro = pares
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    relatorio(pares_primeiro,
              f"NEURAL, BERTimbau multirrótulo (semente {sementes[0]})", ic=True)
    if len(sementes) > 1:
        import statistics
        print(f"  sobre {len(sementes)} sementes: "
              f"micro-F1 {statistics.mean(micros):.3f} ± {statistics.pstdev(micros):.3f} | "
              f"macro-F1 {statistics.mean(macros):.3f} ± {statistics.pstdev(macros):.3f}",
              flush=True)
    print(f"modelo (semente {sementes[0]}) salvo em {SAIDA}/", flush=True)


if __name__ == "__main__":
    main()
