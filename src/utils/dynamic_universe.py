from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


# Ativos com cobertura de dados >= min_coverage dentro da janela.
def eligible_assets(window_df: pd.DataFrame, min_coverage: float = 0.95) -> List[str]:
    if len(window_df) == 0:
        return []
    coverage = window_df.notna().mean()
    return list(coverage[coverage >= min_coverage].index)


# Expande pesos de um subconjunto elegível (d_t) para o vetor de tamanho
def expand_weights(
    w_sub: np.ndarray,
    sub_names: Sequence[str],
    global_names: Sequence[str],
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(global_names)}
    w_full = np.zeros(len(global_names), dtype=float)
    for name, w in zip(sub_names, w_sub):
        pos = idx.get(name)
        if pos is not None:
            w_full[pos] = w
    return w_full


# Retorno do portfólio no dia, robusto a NaN em ativos com peso zero.
def nan_safe_portfolio_return(row_values_full: np.ndarray, w_full: np.ndarray) -> float:
    contrib = np.where(w_full != 0.0, row_values_full * w_full, 0.0)
    return float(np.nansum(contrib))
