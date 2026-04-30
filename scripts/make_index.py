import polars as pl
from pathlib import Path

# ---------- Configurando pastas
project_root = Path(__file__).resolve().parents[1]
data_raw = project_root / 'data/raw'
data_interim = project_root / 'data/interim'
data_processed = project_root / 'data/processed'

# ---------- Carregando dados brutos
pwt_raw = pl.read_csv(data_raw / 'pwt11.csv')

pwt_raw.head()
pwt_raw.shape

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