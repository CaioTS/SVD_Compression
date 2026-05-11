# %%
import polars as pl
import plotly.express as px
from IPython.display import display

# %%
data = pl.read_csv("simulation_results_multiprocessing_filtermem2000.csv")
display(data)

# %%
convtime = data.group_by(
    pl.col("Perturb Freq"),
    pl.col("Is SVD"),
).agg(
    pl.col("Convergence Time").min().alias("Min Convergence Time")
).sort(
    pl.col("Perturb Freq"),
    pl.col("Is SVD"),
)
display(convtime)

# %%
aux = data.select(
    pl.col("Overshoot")
).sort(
    pl.col("Overshoot")
)
px.line(y=aux.to_numpy()[:,0]).show()

# %%
convtimeo = data.filter(
    pl.col("Overshoot") < 5.0
).group_by(
    pl.col("Perturb Freq"),
    pl.col("Is SVD"),
).agg(
    pl.col("Convergence Time").min().alias("Min Convergence Time")
).sort(
    pl.col("Perturb Freq"),
    pl.col("Is SVD"),
)
display(convtimeo)

# %%
display(convtime)
display(convtimeo)
px.line(convtime.to_pandas(), x="Perturb Freq", y="Min Convergence Time", color="Is SVD", title="Convergence Time vs Perturbation Frequency").show()

# %%
display(data)

# %%
convtimemem = data.filter(
    pl.col("Overshoot") < 4.0
).group_by(
    pl.col("Perturb Freq"),
    pl.col("Is SVD"),
    pl.col("Filter Size")
).agg(
    pl.col("Convergence Time").min().alias("Min Convergence Time")
).sort(
    pl.col("Perturb Freq"),
    pl.col("Filter Size"),
    pl.col("Is SVD"),    
)
display(convtimemem)

px.line(convtimemem.to_pandas(), x="Perturb Freq", y="Min Convergence Time", color="Is SVD", line_dash="Filter Size", title="Convergence Time vs Perturbation Frequency by Filter Size").show()

# %%
