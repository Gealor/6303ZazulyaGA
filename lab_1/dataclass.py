from dataclasses import dataclass


@dataclass(slots=True)
class MetObject:
    object_id: str
    classification: str

@dataclass(slots=True)
class ImageObject:
    object_id: str
    primary_image: str