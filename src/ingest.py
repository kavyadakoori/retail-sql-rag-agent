import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.services.ingestion import ingest_documents

    print(f"Ingested {ingest_documents()} chunks")


if __name__ == "__main__":
    main()
