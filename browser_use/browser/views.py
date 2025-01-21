from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel

from browser_use.dom.history_tree_processor.service import DOMHistoryElement
from browser_use.dom.views import DOMState, DOMElementNode, SelectorMap


# Pydantic
class TabInfo(BaseModel):
    """Represents information about a browser tab"""

    page_id: int
    url: str
    title: str


@dataclass
class BrowserState(DOMState):
    url: str
    title: str
    tabs: list[TabInfo]
    screenshot: Optional[str] = None
    pixels_above: int = 0
    pixels_below: int = 0


@dataclass
class BrowserStateHistory(DOMState):
    url: str
    title: str
    tabs: list[TabInfo]
    interacted_element: list[DOMHistoryElement | None] | list[None]
    screenshot: Optional[str] = None
    element_tree: DOMElementNode
    selector_map: SelectorMap
    pixels_above: int = 0
    pixels_below: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {}
        data['tabs'] = [tab.model_dump() for tab in self.tabs]
        data['screenshot'] = self.screenshot
        data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
        data['url'] = self.url
        data['title'] = self.title
        data['element_tree'] = self.element_tree.to_dict()
        data['selector_map'] = self.selector_map.to_dict()
        data['pixels_above'] = self.pixels_above
        data['pixels_below'] = self.pixels_below
        return data


class BrowserError(Exception):
    """Base class for all browser errors"""
