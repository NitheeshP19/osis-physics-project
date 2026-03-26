# app/utils/caching.py
import hashlib
import json
from functools import lru_cache
from typing import Any, Dict

def hash_dict(d: Dict[str, Any]) -> str:
    """Creates a deterministic hash for dictionary parameters."""
    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

# Memory limit: LRU caches drop old items to prevent memory leaks on Render.
@lru_cache(maxsize=1024)
def get_cached_prediction(input_hash: str) -> Dict[str, Any]:
    """Retrieves prediction from cache by hash."""
    # This acts as a placeholder if we want to store raw Python objects.
    # In practice we can just cache the predictor function calls.
    pass

def make_hashable(obj: Any) -> Any:
    """Converts common API payloads into hashable tuples for functools.lru_cache."""
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(i) for i in obj)
    return obj
