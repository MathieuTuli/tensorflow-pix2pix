from pathlib import Path

def file_exists(path: Path) -> bool:
    if not path.exists():
        return False
    return True
