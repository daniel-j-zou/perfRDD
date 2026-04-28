# from data import DataGenConfig, generate_dataset
# from simulations import plot_estimated_vs_true_U_evo
# import matplotlib.pyplot as plt
#
# cfg = DataGenConfig(
#     I0=0.0,
#     I1=0.0,
#     gamma=1.0,
#     theta=0,
#     rho=0,
#     phi=0.0,
#     dgp_id="baseline",
# )
#
# # Pick a sample size and seed you want to inspect
# ds = generate_dataset(n=1000, config=cfg, seed=123)
#
# # Choose c (e.g. 1) matching your simulation setting
# ax = plot_estimated_vs_true_U_evo(ds, c=1.0)
# plt.show()
#
# from data import DataGenConfig, generate_dataset
# from simulations import plot_estimated_vs_true_U_evo, diagnose_U_components
# import matplotlib.pyplot as plt
#
# cfg = DataGenConfig(
#     I0=0.0,
#     I1=0.0,
#     gamma=1.0,
#     theta=1.0,
#     rho=0.3,
#     phi=0.0,
#     dgp_id="baseline",
# )
#
# ds = generate_dataset(n=1000, config=cfg, seed=123)
#
# # First, the simple comparison:
# plot_estimated_vs_true_U_evo(ds, c=1.0)
# plt.show()
#
# # Then, the detailed decomposition:
# diagnose_U_components(ds, c=1.0)
# plt.show()


# from data import DataGenConfig
# from simulations import plot_analytic_vs_mc_U_evo
# import matplotlib.pyplot as plt
#
# cfg = DataGenConfig(
#     I0=0.0,
#     I1=0.0,
#     gamma=1.0,
#     theta=1.0,
#     rho=0.3,
#     phi=0.0,
#     dgp_id="baseline",
# )
#
# ax = plot_analytic_vs_mc_U_evo(
#     config=cfg,
#     c=1.0,
#     N_mc=200000,
#     seed=123,
# )
# plt.show()

from data import DataGenConfig, generate_dataset
from simulations import plot_all_U_flavors_for_dataset
import matplotlib.pyplot as plt

# cfg = DataGenConfig(
#     I0=0.0,
#     I1=0.0,
#     gamma=1.0,
#     theta=1.0,
#     rho=0.3,
#     phi=0.0,
#     dgp_id="baseline",
# )
#
# ds = generate_dataset(n=5000, config=cfg, seed=123)
#
# ax = plot_all_U_flavors_for_dataset(ds, c=1.0)
# plt.show()

cfg_simple = DataGenConfig(
    I0=0.0,
    I1=0.0,
    gamma=1.0,
    theta=0.0,   # no X effect in Y
    rho=0.0,     # no eta effect in Y
    phi=0.0,
    dgp_id="simple",
)

ds_simple = generate_dataset(n=1000, config=cfg_simple, seed=123)
plot_all_U_flavors_for_dataset(ds_simple, c=0, match_mode="one_way")
plt.show()
