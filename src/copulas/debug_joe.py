import logging
logging.basicConfig(level=logging.DEBUG, force=True)
logging.getLogger().setLevel(logging.DEBUG)

import vine_copulas as vc

U, tickers, _ = vc._load_pseudo_obs_from_loader()
print("Ativos:", tickers)

rv = vc.RVineCopula(n_dim=U.shape[1])
rv.fit(
    U,
    families=vc.PRODUCTION_FAMILIES,
    forced_edges=vc.PRODUCTION_FORCED_EDGES,
    tail_err_tol=vc.PRODUCTION_TAIL_ERR_TOL,
    tail_q=vc.PRODUCTION_TAIL_Q,
    n_jobs=1,
)
print("LL:", rv.loglikelihood(), "AIC:", rv.aic(), "sim_order:", rv._sim_order)
