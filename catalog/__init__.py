"""Catalog package: Milvus vectors + upload pipeline (import submodules explicitly to avoid cycles)."""

from catalog.milvus_catalog import rebuild_catalog_from_rows, search_catalog

__all__ = [
    "rebuild_catalog_from_rows",
    "search_catalog",
]
