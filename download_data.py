"""Baixa e valida o dataset público do Kaggle sem exigir credenciais."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://www.kaggle.com/api/v1/datasets.DatasetApiService/DownloadDataset"
DATASET_REQUEST = {
    "ownerSlug": "srisyra02",
    "datasetSlug": "college-major-career-analytics-dataset",
    "datasetVersionNumber": 1,
}
EXPECTED_MEMBER = "college_major_career_outcomes_2026.csv"
EXPECTED_SHA256 = (
    "58f1c10e79793f33049f3b381cd81d4d0c190ca1052e427c29779d0c89c5ceca"
)
DESTINATION = Path("data/raw/college_majors_2026.csv")


def main() -> None:
    if DESTINATION.exists():
        print(f"Dados já existem em {DESTINATION}")
        return
    payload = json.dumps(DATASET_REQUEST).encode("utf-8")
    request = Request(
        URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ann-eda-project/1.0",
        },
    )
    with urlopen(request, timeout=180) as response:
        archive = response.read()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        if EXPECTED_MEMBER not in zf.namelist():
            raise RuntimeError(f"CSV esperado ausente no arquivo: {EXPECTED_MEMBER}")
        temporary = DESTINATION.with_suffix(".csv.part")
        digest = hashlib.sha256()
        with zf.open(EXPECTED_MEMBER) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != EXPECTED_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Hash SHA-256 inesperado para o CSV: "
                f"esperado {EXPECTED_SHA256}, obtido {actual_sha256}"
            )
        temporary.replace(DESTINATION)
    print(f"Dados salvos em {DESTINATION}")


if __name__ == "__main__":
    main()

