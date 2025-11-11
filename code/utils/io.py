from pathlib import Path
import matplotlib.pyplot as plt

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_fig(fig, out_dir: Path, filename: str, dpi: int = 300, tight_layout: bool = True) -> Path:
    ensure_dir(out_dir)
    if tight_layout and hasattr(fig, "tight_layout"):
        try:
            fig.tight_layout()
        except Exception:
            pass
    out_path = out_dir / f"{filename}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
