from pathlib import Path

def save_fig(fig, path, dpi: int = 200, tight: bool = True) -> str:
    """Save a Matplotlib figure to disk, ensuring directories exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)
