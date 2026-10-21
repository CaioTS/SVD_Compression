# %%
import polars as pl
import plotly.express as px
from IPython.display import display

# %%
data = pl.read_csv("simulation_results_multiprocessing_filtermem2000_viga2.csv")
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
convtime.show(limit=None)

#%%
data.filter(
    pl.col("Perturb Freq") == 14.0,
    pl.col("Filter Size") == 2000,
    pl.col("Convergence Time").is_not_null()
).sort(
    pl.col("Convergence Time")
).show(limit=50)

# %%
aux = data.select(
    pl.col("Overshoot")
).sort(
    pl.col("Overshoot")
)
px.line(y=aux.to_numpy()[:,0]).show()

# %%
convtimeo = data.filter(
    pl.col("Overshoot") < 2.0
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
    pl.col("Overshoot") < 2.0
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
convtimemem.show(limit=None)

px.line(convtimemem.to_pandas(), x="Perturb Freq", y="Min Convergence Time", color="Is SVD", 
        line_dash="Filter Size", markers=True,
        title="Convergence Time vs Perturbation Frequency by Filter Size").show()

# %%
print(data.filter(
    pl.col("Filter Size") == 200,
    pl.col("Is SVD") == True,
    pl.col("Perturb Freq") == 14.0,
    pl.col("Convergence Time").is_not_null()
).sort(
    pl.col("Convergence Time")
)
)
# %%
