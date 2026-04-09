import asyncio
import math
from typing import Any, Dict, List

from app.ml.predictor import predict_snr_ber
from app.utils.validation import BatchSimulationItem


def calculate_optimization_score(hybrid_snr_db: float, post_fec_ber: float) -> float:
    """Blend high SNR and low BER into a single ranking score."""
    bounded_ber = max(float(post_fec_ber), 1e-15)
    return float(hybrid_snr_db - (10.0 * math.log10(bounded_ber)))


def _predict_batch_item(item: BatchSimulationItem, fallback_modulation: str) -> Dict[str, Any]:
    config_payload = item.config.model_dump()
    modulation = (item.modulation or fallback_modulation or "OOK-NRZ").upper()
    metrics = predict_snr_ber(dict(config_payload), modulation=modulation)
    hybrid_snr_db = float(metrics["predicted_snr_db"])
    post_fec_ber = float(metrics["estimated_ber"])

    return {
        "clientId": item.clientId,
        "label": item.label or item.clientId,
        "modulation": modulation,
        "inputConfig": config_payload,
        "physics_snr_db": float(metrics["physics_snr_db"]),
        "hybrid_snr_db": hybrid_snr_db,
        "predicted_snr_db": hybrid_snr_db,
        "post_fec_ber": post_fec_ber,
        "estimated_ber": post_fec_ber,
        "optimization_score": calculate_optimization_score(hybrid_snr_db, post_fec_ber),
    }


async def run_batch_simulation(
    batch_configs: List[BatchSimulationItem],
    default_modulation: str = "OOK-NRZ",
) -> Dict[str, Any]:
    prediction_tasks = [
        asyncio.to_thread(_predict_batch_item, item, default_modulation)
        for item in batch_configs
    ]
    ranked_results = await asyncio.gather(*prediction_tasks)
    ranked_results = sorted(
        ranked_results,
        key=lambda item: (
            -float(item["optimization_score"]),
            -float(item["hybrid_snr_db"]),
            float(item["post_fec_ber"]),
        ),
    )

    for rank, item in enumerate(ranked_results, start=1):
        item["rank"] = rank

    best_result = ranked_results[0]
    worst_result = ranked_results[-1]

    return {
        "total_simulations": len(ranked_results),
        "optimal_config": best_result,
        "summary": {
            "best_hybrid_snr_db": best_result["hybrid_snr_db"],
            "lowest_post_fec_ber": min(item["post_fec_ber"] for item in ranked_results),
            "score_spread": float(best_result["optimization_score"] - worst_result["optimization_score"]),
        },
        "rankedResults": ranked_results,
    }
