from dataclasses import asdict, dataclass


@dataclass
class Page:
    document_id: str
    filename: str
    page_number: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    language: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)