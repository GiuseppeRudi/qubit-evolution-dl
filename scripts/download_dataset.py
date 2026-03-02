import urllib.request
from pathlib import Path

URL = "https://github.com/<USER>/<REPO>/releases/download/data-v1/trajectories.csv"
OUT = Path("data/trajectories.csv")

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if OUT.exists():
        print(f"File already exists: {OUT}")
        return

    tmp = OUT.with_suffix(OUT.suffix + ".part")
    print(f"Downloading to {OUT} ...")

    try:
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(OUT)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print("Done.")

if __name__ == "__main__":
    main()