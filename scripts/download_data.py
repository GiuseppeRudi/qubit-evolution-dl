from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


DATASET_URL = (
    "https://github.com/GiuseppeRudi/qubit-evolution-dl/"
    "releases/download/v1.0/trajectories.csv"
)

OUTPUT_PATH = Path("data/trajectories.csv")


def format_mb(bytes_value: int) -> float:
    return bytes_value / (1024 * 1024)


def show_progress(block_count: int, block_size: int, total_size: int) -> None:
    downloaded = block_count * block_size

    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        downloaded_mb = format_mb(downloaded)
        total_mb = format_mb(total_size)

        print(
            f"\rDownloading: {percent:6.2f}% "
            f"({downloaded_mb:.1f} MB / {total_mb:.1f} MB)",
            end="",
            flush=True,
        )
    else:
        downloaded_mb = format_mb(downloaded)
        print(
            f"\rDownloading: {downloaded_mb:.1f} MB",
            end="",
            flush=True,
        )


def download_dataset(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Dataset already exists: {output_path}")
        return

    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    print("Downloading dataset from:")
    print(url)
    print(f"Saving to: {output_path}")

    try:
        urlretrieve(url, temporary_path, reporthook=show_progress)
        temporary_path.replace(output_path)
    except (HTTPError, URLError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Dataset download failed: {error}") from error
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print("\nDataset download completed.")


def main() -> None:
    download_dataset(DATASET_URL, OUTPUT_PATH)


if __name__ == "__main__":
    main()