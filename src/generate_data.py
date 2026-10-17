import numpy as np
import pandas as pd
from config import RAW_DATA_PATH


def generate_helsinki_housing_dataset(
    num_samples: int = 1200, seed: int = 42
) -> pd.DataFrame:
    """Generates a realistic synthetic dataset for Helsinki housing market."""
    np.random.seed(seed)

    districts = {
        "Kamppi": {"base_price_sqm": 8500, "weight": 0.15},
        "Kallio": {"base_price_sqm": 7200, "weight": 0.20},
        "Töölö": {"base_price_sqm": 8100, "weight": 0.15},
        "Pasila": {"base_price_sqm": 5800, "weight": 0.15},
        "Herttoniemi": {"base_price_sqm": 5100, "weight": 0.15},
        "Lauttasaari": {"base_price_sqm": 7600, "weight": 0.10},
        "Malmi": {"base_price_sqm": 4200, "weight": 0.10},
    }

    district_names = list(districts.keys())
    district_weights = [d["weight"] for d in districts.values()]

    selected_districts = np.random.choice(
        district_names, size=num_samples, p=district_weights
    )

    size_sqm = np.random.uniform(22.0, 125.0, size=num_samples).round(1)

    # Estimate room count based on size
    rooms = np.where(
        size_sqm < 35,
        1,
        np.where(
            size_sqm < 55,
            2,
            np.where(size_sqm < 80, 3, np.random.choice([4, 5], size=num_samples)),
        ),
    )

    construction_year = np.random.choice(
        np.arange(1920, 2025),
        size=num_samples,
        p=np.histogram(np.arange(1920, 2025), bins=105)[0]
        / np.histogram(np.arange(1920, 2025), bins=105)[0].sum(),
    )

    floor = np.random.randint(1, 8, size=num_samples)
    has_elevator = np.where(
        floor > 2, np.random.choice([True, False], p=[0.85, 0.15]), False
    )
    has_sauna = np.random.choice([True, False], size=num_samples, p=[0.4, 0.6])
    has_balcony = np.random.choice([True, False], size=num_samples, p=[0.65, 0.35])

    conditions = ["Satisfactory", "Good", "New"]
    condition = np.random.choice(
        conditions, size=num_samples, p=[0.2, 0.65, 0.15]
    )

    # Maintenance fee (~4.5 to 7.0 EUR per sqm)
    monthly_maintenance_fee_eur = (
        size_sqm * np.random.uniform(4.0, 7.5, size=num_samples)
    ).round(2)

    # Calculate target price with domain logic
    prices = []
    for i in range(num_samples):
        dist = selected_districts[i]
        base_sqm = districts[dist]["base_price_sqm"]

        # Condition multiplier
        cond_mult = 0.90 if condition[i] == "Satisfactory" else (1.0 if condition[i] == "Good" else 1.15)

        # Age adjustment
        age = 2026 - construction_year[i]
        age_mult = 1.05 if age < 5 else (0.92 if 20 < age < 50 else 1.0)

        # Feature bonuses
        sauna_bonus = 8000 if has_sauna[i] else 0
        balcony_bonus = 5000 if has_balcony[i] else 0

        sqm_price = base_sqm * cond_mult * age_mult
        total_price = (
            (size_sqm[i] * sqm_price)
            + sauna_bonus
            + balcony_bonus
            + np.random.normal(0, 15000)  # Random market variation
        )

        prices.append(round(max(total_price, 100000), -2))

    df = pd.DataFrame(
        {
            "district": selected_districts,
            "size_sqm": size_sqm,
            "rooms": rooms,
            "construction_year": construction_year,
            "floor": floor,
            "has_elevator": has_elevator,
            "has_sauna": has_sauna,
            "has_balcony": has_balcony,
            "condition": condition,
            "monthly_maintenance_fee_eur": monthly_maintenance_fee_eur,
            "price_eur": prices,
        }
    )

    return df


if __name__ == "__main__":
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_helsinki_housing_dataset()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(
        f"✅ Dataset generated successfully with {len(df)} rows at: {RAW_DATA_PATH}"
    )