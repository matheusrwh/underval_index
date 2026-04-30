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

##########################################
# FIGURAS
##########################################

# ---------- Construindo as visualizações
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
    ax.axhline(0, linestyle='--', color='gray')

fig.supylabel("Índice de desvalorização cambial (log)")

plt.tight_layout()
plt.show()

# ---------- Salvando as visualizações
fig.savefig(figures / 'underval_examples.png', dpi=300)

##########################################
# REGRESSÃO ESTILO RODRIK (2008)
##########################################
pwt = pwt.with_columns(
    (pl.col('ln_gdppc').shift(1).over('iso3')).alias('ln_gdppc_lag')
)

pwt_pandas = pwt.to_pandas().set_index(["iso3", "year"])

X = pwt_pandas[["ln_gdppc_lag", "ln_underval"]]
y = pwt_pandas["gdppc_growth"]

model = PanelOLS(y, X, entity_effects=True, time_effects=True)
res = model.fit(cov_type='clustered', cluster_entity=True)

# ln_gdppc_lag = -0,031*** em Rodrik (2008) com PWT 6.2
# ln_underval = 0,017*** em Rodrik (2008) com PWT 6.2
print(res.summary)