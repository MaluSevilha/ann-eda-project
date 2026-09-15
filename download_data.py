"""Baixa e extrai o dataset público do Kaggle sem exigir credenciais."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai"
)
DESTINATION = Path("data/raw/college_majors_2026.csv")


def main() -> None:
    if DESTINATION.exists():
        print(f"Dados já existem em {DESTINATION}")
        return
    request = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=180) as response:
        archive = response.read()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        member = next(name for name in zf.namelist() if name.endswith(".csv"))
        with zf.open(member) as source, DESTINATION.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    print(f"Dados salvos em {DESTINATION}")


if __name__ == "__main__":
    main()

