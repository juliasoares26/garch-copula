
import numpy as np
import logging
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Funções auxiliares standalone (usadas diretamente pelo EVTCVaROptimizer)

# Σ w_i = 1.
def fully_invested() -> Dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


# Retorno esperado >= target (anualizado).
def min_return_constraint(
    mean_return_fn,
    target: float,
) -> Dict:
    return {"type": "ineq", "fun": lambda w: mean_return_fn(w) - target}


# Σ |w_i - w0_i| / 2 <= max_turnover.
def max_turnover_constraint(
    current_weights: np.ndarray,
    max_turnover: float,
) -> Dict:
    return {
        "type": "ineq",
        "fun": lambda w: max_turnover - np.sum(np.abs(w - current_weights)) / 2.0,
    }


# Σ w_i para ativos do setor <= max_sector_weight.
def sector_weight_constraint(
    sector_indices: List[int],
    max_sector_weight: float,
) -> Dict:
    idx = np.array(sector_indices)
    return {
        "type": "ineq",
        "fun": lambda w: max_sector_weight - np.sum(w[idx]),
    }


# Número de ativos com peso > threshold >= min_assets.
def min_diversification_constraint(
    min_assets: int,
    min_weight_threshold: float = 0.01,
) -> Dict:
    k = 200.0

    def _fun(w: np.ndarray) -> float:
        effective = np.sum(1.0 / (1.0 + np.exp(-k * (w - min_weight_threshold))))
        return effective - min_assets

    return {"type": "ineq", "fun": _fun}


# Classe builder — interface fluente

# Constrói lista de constraints scipy para otimização de portfólio.
class ConstraintBuilder:

    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.d = len(asset_names)
        self._constraints: List[Dict] = []
        self._name_to_idx = {n: i for i, n in enumerate(asset_names)}

    #  Constraints básicos                                                 #

    # Σ w_i = 1.
    def fully_invested(self) -> "ConstraintBuilder":
        self._constraints.append(fully_invested())
        return self

    # w_i <= limit para todo i (equivalente a Bounds, mas via constraint).
    def max_weight(self, limit: float) -> "ConstraintBuilder":
        for i in range(self.d):
            self._constraints.append({
                "type": "ineq",
                "fun": (lambda w, _i=i: limit - w[_i]),
            })
        return self

    # w_i >= limit para todo i.
    def min_weight(
        self,
        limit: float,
        only_invested: bool = False,
    ) -> "ConstraintBuilder":
        for i in range(self.d):
            if only_invested:
                self._constraints.append({
                    "type": "ineq",
                    "fun": (lambda w, _i=i: w[_i] - limit if w[_i] > 1e-6 else 0.0),
                })
            else:
                self._constraints.append({
                    "type": "ineq",
                    "fun": (lambda w, _i=i: w[_i] - limit),
                })
        return self

    # w_i >= 0 (equivalente a Bounds lb=0, mas registrado aqui para rastreamento).
    def long_only(self) -> "ConstraintBuilder":
        for i in range(self.d):
            self._constraints.append({
                "type": "ineq",
                "fun": (lambda w, _i=i: w[_i]),
            })
        return self

    #  Restrições de concentração / setor                                 #

    # Peso total de cada setor <= limit.
    def max_sector_weight(
        self,
        sector_map: Dict[str, List[str]],
        limit: float,
    ) -> "ConstraintBuilder":
        for sector, assets in sector_map.items():
            indices = [self._name_to_idx[a] for a in assets if a in self._name_to_idx]
            if not indices:
                logger.warning(f"Setor '{sector}': nenhum ativo encontrado em asset_names.")
                continue
            self._constraints.append(sector_weight_constraint(indices, limit))
            logger.debug(f"Constraint setor '{sector}': {len(indices)} ativos, limite={limit:.0%}")
        return self

    # Alias legível para max_sector_weight.
    def max_single_sector_concentration(
        self,
        sector_map: Dict[str, List[str]],
        max_fraction_of_portfolio: float = 0.50,
    ) -> "ConstraintBuilder":
        return self.max_sector_weight(sector_map, max_fraction_of_portfolio)

    #  Restrições de turnover / custos                                     #

    # Σ |w_i - w0_i| / 2 <= max_turnover.
    def max_turnover(
        self,
        current_weights: np.ndarray,
        max_turnover: float,
    ) -> "ConstraintBuilder":
        w0 = np.asarray(current_weights, dtype=float)
        self._constraints.append(max_turnover_constraint(w0, max_turnover))
        return self

    #  Restrições de retorno / risco                                       #

    # Retorno esperado anualizado >= target.
    def min_expected_return(
        self,
        mean_return_fn,
        target: float,
    ) -> "ConstraintBuilder":
        self._constraints.append(min_return_constraint(mean_return_fn, target))
        return self

    # Número mínimo de ativos com peso relevante (>= threshold).
    def min_diversification(
        self,
        min_assets: int,
        min_weight_threshold: float = 0.01,
    ) -> "ConstraintBuilder":
        self._constraints.append(
            min_diversification_constraint(min_assets, min_weight_threshold)
        )
        return self

    #  Restrições de leverage                                             #

    # Σ |w_i| <= limit (relevante quando allow_short=True).
    def max_gross_leverage(self, limit: float = 1.0) -> "ConstraintBuilder":
        self._constraints.append({
            "type": "ineq",
            "fun": lambda w: limit - np.sum(np.abs(w)),
        })
        return self

    # Σ w_i^- <= max_short (posição short total).
    def max_net_short(self, max_short: float = 0.30) -> "ConstraintBuilder":
        self._constraints.append({
            "type": "ineq",
            "fun": lambda w: max_short - np.sum(np.abs(np.minimum(w, 0))),
        })
        return self

    #  Build                                                               #

    # Retorna lista de constraints para scipy.optimize.minimize.
    def build(self) -> List[Dict]:
        logger.info(f"ConstraintBuilder: {len(self._constraints)} constraints construídas.")
        return list(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)

    def __repr__(self) -> str:
        return f"ConstraintBuilder(d={self.d}, n_constraints={len(self._constraints)})"


# Presets prontos para uso rápido

# Preset padrão long-only:
def standard_long_only_constraints(
    asset_names: List[str],
    max_weight: float = 0.40,
    min_weight: float = 0.0,
) -> List[Dict]:
    return (
        ConstraintBuilder(asset_names)
        .fully_invested()
        .max_weight(max_weight)
        .build()
    )


# Preset para uso institucional com restrições de setor e turnover.
def institutional_constraints(
    asset_names: List[str],
    sector_map: Optional[Dict[str, List[str]]] = None,
    max_weight: float = 0.30,
    max_sector: float = 0.40,
    max_turnover: float = 0.25,
    current_weights: Optional[np.ndarray] = None,
    min_return_fn=None,
    min_return_target: float = 0.0,
) -> List[Dict]:
    builder = ConstraintBuilder(asset_names).fully_invested().max_weight(max_weight)

    if sector_map:
        builder.max_sector_weight(sector_map, max_sector)

    if current_weights is not None:
        builder.max_turnover(current_weights, max_turnover)

    if min_return_fn is not None:
        builder.min_expected_return(min_return_fn, min_return_target)

    return builder.build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    tickers = ["ITUB4", "BBDC4", "PETR4", "VALE3", "EGIE3", "SUZB3"]
    d = len(tickers)
    w0 = np.ones(d) / d

    sector_map = {
        "Financeiro": ["ITUB4", "BBDC4"],
        "Commodities": ["PETR4", "VALE3"],
        "Utilities": ["EGIE3"],
        "Papel": ["SUZB3"],
    }

    cons_simple = standard_long_only_constraints(tickers, max_weight=0.35)
    print(f"Constraints long-only: {len(cons_simple)}")

    cons_inst = institutional_constraints(
        tickers,
        sector_map=sector_map,
        max_weight=0.30,
        max_sector=0.50,
        max_turnover=0.20,
        current_weights=w0,
    )
    print(f"Constraints institucionais: {len(cons_inst)}")

    for i, c in enumerate(cons_inst):
        val = c["fun"](w0)
        sat = "OK" if (val >= 0 if c["type"] == "ineq" else abs(val) < 1e-8) else "VIOLADO"
        print(f"  Constraint {i}: val={val:.6f}  [{sat}]")

    cons_custom = (
        ConstraintBuilder(tickers)
        .fully_invested()
        .max_weight(0.30)
        .min_weight(0.02)
        .max_sector_weight(sector_map, 0.45)
        .max_turnover(w0, 0.15)
        .min_diversification(min_assets=4)
        .build()
    )
    print(f"\nConstraints custom (builder fluente): {len(cons_custom)}")
    print("\nTodos os testes concluídos.")
