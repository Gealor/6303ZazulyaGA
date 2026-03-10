from dataclasses import dataclass


@dataclass
class BaseObject:
    object_id: str


@dataclass(slots=True)
class MetObject(BaseObject):
    classification: str


@dataclass(slots=True)
class ImageObject(BaseObject):
    object_id: str
    primary_image: str
