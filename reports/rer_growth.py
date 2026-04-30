# -------------------------------------------
# Associações entre desvalorização cambial e crescimento.
# Figuras exemplicativas e análises de regressão.
#
# Autor: Matheus Rosa
# Data: 29-04-2026
# Fonte: PWT 11.0
# -------------------------------------------

import polars as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from pathlib import Path

# ---------- Configurando pastas
project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / 'data/raw'
data_interim = project_root / 'data/interim'
data_processed = project_root / 'data/processed'
figures = project_root / 'reports/figures'

# ---------- Carregando dados
pwt = pl.read_csv(data_interim / 'underval_index.csv')

# ---------- Construindo visualizações
countries = ["BRA", "CHN", "IND", "MEX"]
line_color = ['maroon']

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

df = pwt.filter(pl.col("iso3") == countries[0])
axs[0, 0].plot(df["year"], df["ln_underval"],
               label=countries[0], color = line_color[0], linewidth=2)

df = pwt.filter(pl.col("iso3") == countries[1])
axs[0, 1].plot(df["year"], df["ln_underval"],
               label=countries[1], color = line_color[0], linewidth=2)

df = pwt.filter(pl.col("iso3") == countries[2])
axs[1, 0].plot(df["year"], df["ln_underval"],
               label=countries[2], color = line_color[0], linewidth=2)

df = pwt.filter(pl.col("iso3") == countries[3])
axs[1, 1].plot(df["year"], df["ln_underval"],
               label=countries[3], color = line_color[0], linewidth=2)

for ax in axs.flat:
    ax.set_title(f"{ax.get_lines()[0].get_label()}")
    ax.axhline(0, linestyle='--', linewidth=0.7, color='gray')

fig.supylabel("Índice de desvalorização cambial (log)")

plt.tight_layout()
plt.show()

# ---------- Salvando as visualizações
fig.savefig(figures / 'underval_examples.png', dpi=300)
