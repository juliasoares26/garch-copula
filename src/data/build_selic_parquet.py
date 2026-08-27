from __future__ import annotations

import argparse
from pathlib import Path

from risk_free_loader import load_risk_free_rate, _DEFAULT_DATA_DIR


# Lê a série via risk_free_loader e salva em formato compatível com
def build_parquet(source: str, raw_data_dir: Path, external_dir: Path) -> Path:
    df = load_risk_free_rate(source=source, data_dir=raw_data_dir)

    out = df[["Rate_aa"]].rename(columns={"Rate_aa": "taxa_anual"})

    external_dir.mkdir(parents=True, exist_ok=True)
    out_path = external_dir / f"{source}.parquet"
    out.to_parquet(out_path)

    print(f"[{source.upper()}] {len(out)} obs salvas em {out_path}")
    print(f"  período completo salvo: {out.index.min().date()} -> {out.index.max().date()}")
    recent = out[out.index >= "2020-01-01"]
    if len(recent) > 0:
        print(f"  taxa_anual média (2020+): {recent['taxa_anual'].mean()*100:.2f}%  "
              f"última: {recent['taxa_anual'].iloc[-1]*100:.2f}%")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["selic", "cdi", "both"], default="both",
        help="qual(is) série(s) gerar (default: both)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="diretório 'data/' raiz do projeto (default: detectado a partir "
             "de risk_free_loader.py, ou utils.config se disponível)",
    )
    args = parser.parse_args()

    raw_data_dir = Path(args.data_dir) / "raw" if args.data_dir else _DEFAULT_DATA_DIR
    external_dir = (
        Path(args.data_dir) / "external" if args.data_dir
        else raw_data_dir.parent / "external"
    )

    sources = ["selic", "cdi"] if args.source == "both" else [args.source]
    for src in sources:
        try:
            build_parquet(src, raw_data_dir, external_dir)
        except FileNotFoundError as e:
            print(f"[{src.upper()}] PULADO — {e}")


if __name__ == "__main__":
    main()
