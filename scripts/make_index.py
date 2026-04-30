# -------------------------------------------
# Estimativa do índice de desvalrização cambial
# seguindo o procedimento de Rodrik (2008).
# Autor: Matheus Rosa
# Data: 29-04-2026
# Fonte: PWT 11.0
# -------------------------------------------

import polars as pl
import statsmodels.api as sm
from linearmodels import PanelOLS
from pathlib import Path

# ---------- Configurando pastas
project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / 'data/raw'
data_interim = project_root / 'data/interim'
data_processed = project_root / 'data/processed'

# ---------- Carregando dados brutos
pwt_raw = pl.read_csv(data_raw / 'pwt11.csv')

# ---------- Tratando os dados
pwt_raw = pwt_raw.drop_nulls(subset=["pl_gdpo", "pop", "rgdpe", "xr"])

pwt_raw = pwt_raw.filter(
    (pl.col("pl_gdpo") > 0) &
    (pl.col("pop") > 0) &
    (pl.col("rgdpe") > 0) &
    (pl.col("xr") > 0)
)

# ---------- Organizando variáveis básicas
pwt_raw = pwt_raw.with_columns(
    (pl.col("pl_gdpo") * pl.col("xr")).alias("ppp"),
    (pl.col("xr") / (pl.col("pl_gdpo") * pl.col("xr"))).alias("rer"),
    (pl.col("xr") / (pl.col("pl_gdpo") * pl.col("xr"))).log().alias("ln_rer"),
    (pl.col("rgdpe") / pl.col("pop")).alias("gdppc"),
    (pl.col("rgdpe") / pl.col("pop")).log().alias("ln_gdppc"),
)

pwt_raw = pwt_raw.with_columns(
    ((pl.col("ln_gdppc")) - (pl.col("ln_gdppc").shift(1).over("iso3"))).alias("gdppc_growth")
)

pwt = pwt_raw.select(["iso3", "Country", "year", "ln_rer", "ln_gdppc", "gdppc_growth"])

pwt.head()

# ---------- Correção do efeito Balassa-Samuelson
pwt_pandas = pwt.to_pandas().set_index(["iso3", "year"])

y = pwt_pandas["ln_rer"]
X = pwt_pandas["ln_gdppc"]

X = sm.add_constant(X)

model = PanelOLS(y, X, entity_effects=False, time_effects=True)
res = model.fit(cov_type='clustered', cluster_entity=True)

# \beta = -0,240 em Rodrik (2008) com PWT 6.2
print(res.summary)

predict = res.predict(X)

pwt_pandas['ln_rer_hat'] = predict

pwt = pl.from_pandas(pwt_pandas.reset_index())

# ---------- Índice de desvalorização cambial
pwt = pwt.with_columns(
    (pl.col("ln_rer") - pl.col("ln_rer_hat")).alias("ln_underval")
)

pwt.head()

# ---------- Salvando dados intermediários
pwt.write_excel(data_interim / 'underval_index.xlsx')