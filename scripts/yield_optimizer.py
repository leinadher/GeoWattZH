from scripts.predict_energy import predict_energy_yield
import numpy as np

def optimize_yield(features):
    """
    Finds the optimal configuration of probe count and depth (within max limit),
    maximizing yield per probe while discouraging oversized systems via a soft penalty.
    """
    depth_max = int(features["depth_max"])
    depth_range = range(10, depth_max + 1, 5)
    probe_range = range(1, 11)

    best_config = {
        "yield": -np.inf,
        "efficiency": -np.inf  # Here 'efficiency' refers to the score including penalty
    }

    for depth in depth_range:
        for probes in probe_range:
            bottom_elevation = features.get("elevation", 0) - depth
            inputs = {
                "Gesamtsondenzahl": probes,
                "count_100m": features.get("count_100m", 0),
                "nearest_borehole_dist": features.get("nearest_borehole_dist", 0),
                "Sondentiefe": depth,
                "bottom_elevation": bottom_elevation
            }

            pred = predict_energy_yield(inputs)

            # Reward yield per probe, adjusted by log scale
            base_efficiency = (pred / probes) * np.log1p(probes)

            # Discourages large systems (non-linear)
            size_penalty = (probes / 10) ** 2 + (depth / 200) ** 2

            score = base_efficiency / (1 + size_penalty)

            if score > best_config["efficiency"]:
                best_config.update({
                    "yield": pred,
                    "depth": depth,
                    "probes": probes,
                    "efficiency": score
                })

    return best_config