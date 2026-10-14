import logging
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
# Sans "www" : avec, le site répond par une redirection vers cette adresse.
URL = "https://football-data.co.uk/mmz4281/{code}/F1.csv"
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2017, 2027)]

# PSC* = cotes de clôture Pinnacle (H = domicile, D = nul, A = extérieur).
COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSCH", "PSCD", "PSCA"]


def download_seasons() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for code in SEASONS:
        try:
            urllib.request.urlretrieve(URL.format(code=code), RAW_DIR / f"F1_{code}.csv")
        except urllib.error.URLError:
            logger.exception("Téléchargement échoué pour la saison %s", code)


def load_matches() -> pd.DataFrame:
    frames = []
    for code in SEASONS:
        # utf-8-sig : le CSV commence par un BOM qui colle au nom de la 1re colonne.
        # Colonnes choisies par nom : leur position change d'une saison à l'autre.
        df = pd.read_csv(
            RAW_DIR / f"F1_{code}.csv", encoding="utf-8-sig", usecols=lambda c: c in COLUMNS
        )
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Saison %s : colonnes absentes, mises à NaN : %s", code, missing)
        df = df.reindex(columns=COLUMNS)
        # 2017-18 écrit l'année sur 2 chiffres (04/08/17), les autres sur 4.
        date_format = "%d/%m/%y" if len(df["Date"].iloc[0]) == 8 else "%d/%m/%Y"
        df["Date"] = pd.to_datetime(df["Date"], format=date_format)
        df["season"] = code
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    # Tri chronologique avant tout calcul (règle anti-fuite) ; stable pour que
    # deux matchs du même jour gardent leur ordre d'origine.
    return matches.sort_values("Date", kind="stable").reset_index(drop=True)
