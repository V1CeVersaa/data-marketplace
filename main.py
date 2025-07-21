import logging

import numpy as np
from tqdm import tqdm

from src.market import Marketplace, generate_synthetic_market
from src.model import MLModel
from src.pricer import MWUPricer


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    sellers, buyers = generate_synthetic_market()

    pricer = MWUPricer(
        b_min=0.0,
        b_max=2.0,
        N=len(buyers),
        L_lipschitz=1.0,
    )

    mp = Marketplace(
        sellers=sellers,
        pricer=pricer,
        model_factory=MLModel,
        noise_sigma=1.0,
        shapley_samples=100,  # Reduced for faster demo
        robust_shapley=True,
    )

    history: list[dict[str, float]] = []
    for buyer in tqdm(buyers, desc="Running market simulation"):
        info = mp.transact(buyer)
        history.append(info)

    revs = [h["revenue"] for h in history]
    print("\n====== Simulation Finished ======")
    print(f"Total Buyers      : {len(buyers)}")
    print(f"Total Revenue     : {np.sum(revs):.4f}")
    print(f"Average Revenue   : {np.mean(revs):.4f}")

    sellers_sorted = sorted(mp.sellers, key=lambda s: s.revenue, reverse=True)
    print("\nTop 5 Sellers by Revenue:")
    for s in sellers_sorted[:5]:
        print(f"  Seller {s.idx:02d} | Revenue = {s.revenue:.4f}")


if __name__ == "__main__":
    main()
