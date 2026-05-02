# -------------------------------------------
# Associações entre desvalorização cambial e crescimento.
# Figuras exemplicativas e análises de regressão.
#
# Autor: Matheus Rosa
# Data: 29-04-2026
# Fonte: PWT 11.0
# -------------------------------------------
import polars as pl
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from linearmodels import PanelOLS
from pathlib import Path

# ---------- Configurando pastas
project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / 'data/raw'
data_interim = project_root / 'data/interim'
data_processed = project_root / 'data/processed'
figures = project_root / 'reports/figures'
models = project_root / 'reports/models'

# ---------- Carregando dados
pwt = pl.read_csv(data_interim / 'underval_index.csv')

##########################################
# FIGURAS
##########################################
# ---------- Histograma alisado do índice de desvalorização cambial
plt.figure(figsize=(10, 6))

pwt = pwt.drop_nulls(subset=["ln_underval"])

plt.hist(pwt["ln_underval"], bins=30, density=True, alpha=0.6, color='maroon')
plt.xlabel("Índice de desvalorização cambial (log)")
plt.ylabel("Densidade")

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------- Trajetórias do índice de desvalorização em subplots
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

fig.savefig(figures / 'underval_examples.png', dpi=300)

# ---------- Dispersão entre desvalorização cambial e crescimento
fig_disp = plt.figure(figsize=(10, 6))

pwt = pwt.drop_nulls(subset=["ln_underval", "gdppc_growth"])

plt.scatter(pwt["ln_underval"], pwt["gdppc_growth"], alpha=0.5, color='maroon')

fit = sm.ols("gdppc_growth ~ ln_underval", data=pwt.to_pandas()).fit()
plt.plot(pwt["ln_underval"], fit.fittedvalues,
         linestyle='--', linewidth=2, color='black')

plt.xlabel("Índice de desvalorização cambial (log)")
plt.ylabel("Crescimento do PIB per capita")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

fig_disp.savefig(figures / 'underval_scatter.png', dpi=300)

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
with open(models / "rodrik_style_regression.txt", "w") as f:
    f.write(res.summary.as_text())