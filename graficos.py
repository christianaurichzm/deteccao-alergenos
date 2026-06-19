"""
Gera as figuras principais do estudo a partir dos resultados medidos
(RESULTADOS.md, snapshot de agosto/2023) e salva em figuras/.

Uso:
    python3 graficos.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

SAIDA = "figuras"

virgula = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def robustez(caminho):
    taxas = [0, 5, 10, 15, 20]
    simbolico = [0.868, 0.799, 0.733, 0.674, 0.603]
    neural = [0.792, 0.767, 0.744, 0.706, 0.666]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(taxas, simbolico, "o-", color="#1f6feb", label="Simbólico (Ontologia + Regras)")
    ax.plot(taxas, neural, "s-", color="#d1242f", label="Neural (BERTimbau)")
    ax.axvspan(10, 20, color="#d1242f", alpha=0.06)
    ax.annotate("neural mais robusto\na partir de ~10%", xy=(15, 0.706),
                xytext=(12.5, 0.60), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("Taxa de ruído de OCR (%)")
    ax.set_ylabel("micro-F1")
    ax.set_title("Robustez a ruído de OCR: simbólico vs. neural")
    ax.set_xticks(taxas)
    ax.yaxis.set_major_formatter(virgula)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def curva_aprendizado(caminho):
    n = [239, 598, 1196, 1794, 2393]
    micro = [0.541, 0.647, 0.716, 0.797, 0.827]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n, micro, "o-", color="#1f6feb", label="TF-IDF + Reg. Logística")
    ax.axhline(0.868, ls="--", color="#2da44e",
               label="Simbólico (Ontologia + Regras) = 0,868")
    ax.annotate("ainda sem platô", xy=(2393, 0.827), xytext=(1450, 0.775),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("Exemplos de treino")
    ax.set_ylabel("micro-F1")
    ax.set_title("Curva de aprendizado: desempenho limitado por dados")
    ax.yaxis.set_major_formatter(virgula)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def comparacao(caminho):
    nomes = ["BASE (substring)", "Ontologia (OWL/SPARQL)", "Ontologia + Regras",
             "Ontologia + BERT (WSD)", "TF-IDF + Reg. Logística",
             "BERTimbau multirrótulo", "BERTimbau-large + LoRA"]
    micro = [0.773, 0.800, 0.868, 0.868, 0.827, 0.788, 0.750]
    ic_lo = [0.748, 0.779, 0.849, 0.847, 0.807, 0.764, 0.724]
    ic_hi = [0.797, 0.822, 0.887, 0.886, 0.848, 0.814, 0.778]
    cores = ["#1f6feb", "#1f6feb", "#1f6feb", "#8250df",
             "#bf8700", "#bf8700", "#bf8700"]

    err = [[m - lo for m, lo in zip(micro, ic_lo)],
           [hi - m for m, hi in zip(micro, ic_hi)]]
    y = list(range(len(nomes)))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y, micro, xerr=err, color=cores, alpha=0.85,
            error_kw=dict(ecolor="#333", capsize=3))
    for yi, hi in zip(y, ic_hi):
        ax.text(hi + 0.004, yi, f"{micro[yi]:.3f}".replace(".", ","),
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.invert_yaxis()
    ax.set_xlim(0.70, 0.95)
    ax.set_xlabel("micro-F1 (barras de erro: IC95% bootstrap)")
    ax.set_title("Comparação dos 7 sistemas no conjunto de teste")
    ax.xaxis.set_major_formatter(virgula)
    ax.grid(True, axis="x", alpha=0.3)
    legenda = [Patch(color="#1f6feb", label="Simbólico"),
               Patch(color="#8250df", label="Híbrido"),
               Patch(color="#bf8700", label="Supervisionado")]
    ax.legend(handles=legenda, loc="lower right", fontsize=9)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(SAIDA, exist_ok=True)
    robustez(f"{SAIDA}/robustez_ocr.png")
    curva_aprendizado(f"{SAIDA}/curva_aprendizado.png")
    comparacao(f"{SAIDA}/comparacao_sistemas.png")
    print(f"figuras salvas em {SAIDA}/")


if __name__ == "__main__":
    main()
