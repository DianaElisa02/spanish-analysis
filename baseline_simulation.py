import pandas as pd
import numpy as np
import random
import logging
import unicodedata

np.random.seed(42)
random.seed(42)
logging.basicConfig(level=logging.INFO)


def load_eff_data(file_path="eff_data.xlsx"):
    eff_df = pd.read_excel(file_path, sheet_name="Datos", skiprows=10)
    eff_df.columns = ["Concept", "Element", "Statistic", "Breakdown", "Category", "Measure", "Wave", "Value"]
    eff_df["Wave"] = pd.to_numeric(eff_df["Wave"], errors="coerce")
    eff_df["Value"] = pd.to_numeric(eff_df["Value"], errors="coerce")
    eff_df["Value"] *= 1000
    eff_df["Category"] = eff_df["Category"].astype(str).str.strip().str.replace("\u2013", "-", regex=False)

    filtered = eff_df[
        (eff_df["Wave"] == 2022) &
        (eff_df["Breakdown"] == "NET WEALTH PERCENTILE") &
        (eff_df["Statistic"].str.upper() == "MEAN") &
        (eff_df["Value"].notna())
        ].copy()
    # Filter to valid wealth groups only
    valid_categories = [
        "under 25", "between 25 and 50",
        "between 50 and 75", "between 75 and 90",
        "between 90 and 100"
    ]
    filtered["Category"] = filtered["Category"].str.lower().str.strip()
    filtered = filtered[filtered["Category"].isin(valid_categories)]

    return filtered


def process_eff_assets_income(filtered_df):
    real_assets = [
        "MAIN RESIDENCE", "OTHER REAL ESTATE PROPERTIES",
        "CARS AND OTHER VEHICLES", "OTHER DURABLE GOODS"
    ]
    financial_assets = [
        "LISTED SHARES", "INVESTMENT FUNDS", "FIXED-INCOME SECURITIES",
        "PENSION SCHEMES AND UNIT-LINKED OR MIXED LIFE INSURANCE",
        "ACCOUNTS AND DEPOSITS USABLE FOR PAYMENTS",
        "ACCOUNTS NON USABLE FOR PAYMENTS AND HOUSE-PURCHASE SAVING ACCOUNTS",
        "OTHER FINANCIAL ASSETS", "UNLISTED SHARES AND OTHER EQUITY", "TOTAL REAL ASSETS"
    ]
    debts = ["TOTAL DEBT"]

    expected_columns = set(real_assets + financial_assets + debts)
    if not expected_columns.issubset(set(filtered_df["Element"].unique())):
        missing = expected_columns - set(filtered_df["Element"].unique())
        raise ValueError(f"Missing expected elements in data: {missing}")

    pivot_df = filtered_df.pivot_table(index="Category", columns="Element", values="Value", aggfunc="mean").fillna(0)

    pivot_df["Real_Assets"] = pivot_df[real_assets].sum(axis=1)
    pivot_df["Financial_Assets"] = pivot_df[financial_assets].sum(axis=1)
    pivot_df["Total_Assets"] = pivot_df["Real_Assets"] + pivot_df["Financial_Assets"]
    pivot_df["Debts"] = pivot_df[debts].sum(axis=1)
    pivot_df["Net_Wealth"] = pivot_df["Total_Assets"] - pivot_df["Debts"]

    # Avoid division by zero
    pivot_df["Real_Asset_Ratio"] = pivot_df["Real_Assets"] / pivot_df["Total_Assets"].replace(0, np.nan)
    pivot_df["Financial_Asset_Ratio"] = pivot_df["Financial_Assets"] / pivot_df["Total_Assets"].replace(0, np.nan)
    pivot_df["Debt_Ratio"] = pivot_df["Debts"] / pivot_df["Total_Assets"].replace(0, np.nan)

    income_df = filtered_df[filtered_df["Concept"].str.upper().str.contains("INCOME")][["Category", "Value"]].copy()
    income_df.columns = ["Category", "Mean_Income"]
    income_df["Mean_Income"] *= 1000

    business_df = filtered_df[filtered_df["Element"] == "BUSINESSES RELATED TO SELF-EMPLOYMENT"][
        ["Category", "Value"]].copy()
    business_df.columns = ["Category", "Business_Assets"]

    pivot_df = pivot_df.reset_index()
    pivot_df = pivot_df.merge(income_df, on="Category", how="left")
    pivot_df = pivot_df.merge(business_df, on="Category", how="left")
    pivot_df["Business_Assets"] = pivot_df["Business_Assets"].fillna(0.0)
    pivot_df["Business_Asset_Ratio"] = pivot_df["Business_Assets"] / pivot_df["Total_Assets"].replace(0, np.nan)
    pivot_df["Category"] = pivot_df["Category"].astype(str).str.strip().str.lower()

    group_stats_df = pivot_df.drop_duplicates(subset="Category")[[
        "Category", "Total_Assets", "Debts", "Net_Wealth",
        "Real_Asset_Ratio", "Financial_Asset_Ratio", "Debt_Ratio",
        "Mean_Income", "Business_Assets", "Business_Asset_Ratio"
    ]]

    return pivot_df, group_stats_df


def reweight_to_match_percentile_shares(
    dataframe,
    value_col="Net_Wealth",
    weight_col="Weight",
    percentiles=10
):
    df_copy = dataframe.copy()

    # Auto-calculate Net_Wealth if missing
    if value_col not in df_copy.columns:
        if "Total_Assets" in df_copy.columns and "Debts" in df_copy.columns:
            print(f"ℹ️ {value_col} not found. Auto-calculating it as Total_Assets - Debts.")
            df_copy[value_col] = df_copy["Total_Assets"] - df_copy["Debts"]
        else:
            raise ValueError(f"❌ Cannot compute {value_col}. Missing Total_Assets or Debts.")

    # Ensure weights are valid
    if weight_col not in df_copy.columns:
        raise ValueError(f"❌ Weight column '{weight_col}' not found in DataFrame.")

    df_copy = df_copy[df_copy[value_col] >= 0].reset_index(drop=True)

    df_copy["Wealth_Rank"] = df_copy[value_col].rank(method="first", pct=True)
    df_copy["Wealth_Percentile"] = pd.qcut(df_copy["Wealth_Rank"], q=percentiles, labels=False)
    df_copy["Weighted_Wealth"] = df_copy[value_col] * df_copy[weight_col]

    actual_shares = df_copy.groupby("Wealth_Percentile")["Weighted_Wealth"].sum()
    actual_shares /= actual_shares.sum()

    target_shares = np.array([0.00, 0.01, 0.02, 0.04, 0.07,
                              0.10, 0.13, 0.18, 0.25, 0.20])
    target_shares /= target_shares.sum()

    if len(actual_shares) != len(target_shares):
        raise ValueError("❌ Mismatch in number of percentiles vs target shares.")

    scaling_factors = target_shares / actual_shares.values
    df_copy["Scaling_Factor"] = df_copy["Wealth_Percentile"].map(dict(enumerate(scaling_factors)))
    df_copy["Adjusted_Weight"] = df_copy[weight_col] * df_copy["Scaling_Factor"]

    return df_copy

def calculate_population_over_30(pop_file):
    df = pd.read_csv(pop_file)
    df["Region"] = df["Region"].str.replace(r"^\d+\s+", "", regex=True)
    df["Region"] = df["Region"].apply(
        lambda x: unicodedata.normalize("NFKD", x.strip()).encode("ascii", errors="ignore").decode("utf-8").lower()
    )

    province_to_region = {
        "madrid": "madrid", "madrid, comunidad de": "madrid",
        "barcelona": "catalonia", "girona": "catalonia", "lleida": "catalonia", "tarragona": "catalonia", "cataluna": "catalonia",
        "valencia/valencia": "valencia", "alicante/alacant": "valencia", "castellon/castello": "valencia", "comunitat valenciana": "valencia",
        "coruna, a": "galicia", "lugo": "galicia", "ourense": "galicia", "pontevedra": "galicia",
        "asturias, principado de": "asturias", "asturias": "asturias",
        "caceres": "extremadura", "badajoz": "extremadura"
    }

    df["Autonomous_Region"] = df["Region"].map(province_to_region)
    df = df[df["Autonomous_Region"].notna()].copy()

    over_30_bins = [
        "30-34", "35-39", "40-44", "45-49", "50-54",
        "55-59", "60-64", "65+"
    ]
    df = df[df["Age Bin"].isin(over_30_bins)]

    region_pop = df.groupby("Autonomous_Region")["Population"].sum()
    selected_regions = ["madrid", "catalonia", "valencia", "galicia", "asturias", "extremadura"]
    total_population = region_pop.loc[region_pop.index.intersection(selected_regions)].sum()

    print(f" Estimated population over 30 in selected regions: {total_population:,.0f}")
    return total_population

def load_population_and_revenue_data(pop_file):
    observed_df = pd.read_csv("Cleaned_Regional_Wealth_Tax_Data.csv")
    observed_clean = observed_df[
        observed_df["Variable"].str.strip().str.lower() == "resultado de la declaración"
    ].copy()
    observed_clean["Region"] = observed_clean["Region"].str.strip().str.lower()
    observed_clean = observed_clean.rename(columns={"Importe": "Total_Revenue"})
    revenue_df = observed_clean[["Region", "Total_Revenue"]].copy()

    pop_shares = pd.read_csv(pop_file)
    pop_shares["Region"] = pop_shares["Region"].str.replace(r"^\d+\s+", "", regex=True)
    pop_shares["Region"] = pop_shares["Region"].apply(
        lambda x: unicodedata.normalize("NFKD", x.strip()).encode("ascii", errors="ignore").decode("utf-8").lower()
    )

    province_to_region = {
        "madrid": "madrid", "madrid, comunidad de": "madrid",
        "barcelona": "catalonia", "girona": "catalonia", "lleida": "catalonia", "tarragona": "catalonia", "cataluna": "catalonia",
        "valencia/valencia": "valencia", "alicante/alacant": "valencia", "castellon/castello": "valencia", "comunitat valenciana": "valencia",
        "coruna, a": "galicia", "lugo": "galicia", "ourense": "galicia", "pontevedra": "galicia",
        "asturias, principado de": "asturias", "asturias": "asturias",
        "caceres": "extremadura", "badajoz": "extremadura"
    }

    pop_shares["Autonomous_Region"] = pop_shares["Region"].map(province_to_region)
    dropped = pop_shares["Autonomous_Region"].isna().sum()
    if dropped > 0:
        print(f"{dropped} rows dropped due to unmatched province mapping.")
    pop_shares = pop_shares[pop_shares["Autonomous_Region"].notna()].copy()

    region_population = pop_shares.groupby("Autonomous_Region", as_index=False)["Population"].sum()
    region_population.columns = ["Region", "Population"]
    region_population["Population"] = region_population["Population"] / region_population["Population"].sum()

    return revenue_df, region_population

def compute_region_targets(region_weights, total_households):
    region_targets = {
        row["Region"]: int(round(row["Population"] * total_households))
        for _, row in region_weights.iterrows()
    }
    print("Computed household targets by region:")
    for region, count in region_targets.items():
        print(f"  {region}: {count} households")
    return region_targets

def generate_scaled_households(region_weights_df, simulated_population, avg_household_size=1.6, rng_seed=42):
    """
    Generates individual household rows with region and realistic household sizes
    for a simulated adult population (e.g., aged 30+).

    Parameters:
        region_weights_df (pd.DataFrame): Must contain 'Region' and 'Population' (as shares summing to 1).
        simulated_population (int): Total number of adults in the synthetic population.
        avg_household_size (float): Expected average size of households (default = 1.6 for 30+).
        rng_seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Rows with columns ['Region', 'Household_Size'].
    """
    rng = np.random.default_rng(rng_seed)
    df = region_weights_df.copy()

    # 1. Calculate number of households per region
    total_households = int(round(simulated_population / avg_household_size))
    df["Num_Households"] = (df["Population"] * total_households).round().astype(int)

    # Adjust for rounding mismatch
    diff = total_households - df["Num_Households"].sum()
    if diff != 0:
        idx = df["Num_Households"].idxmax()
        df.loc[idx, "Num_Households"] += diff

    # 2. Generate household sizes (1 or 2 members)
    regions = np.repeat(df["Region"].values, df["Num_Households"].values)
    household_sizes = rng.choice([1, 2], size=total_households, p=[0.3, 0.7])

    # 3. Create final DataFrame
    household_df = pd.DataFrame({
        "Region": regions,
        "Household_Size": household_sizes
    })

    print(f"✅ Generated {len(household_df)} households.")
    return household_df

def share_in_top_percentile(df, value_col, weight_col, top_pct=0.01):
    df = df[df[value_col] > 0].copy()
    df = df.sort_values(by=value_col, ascending=False)

    total_weight = df[weight_col].sum()
    if total_weight == 0:
        return 0.0

    df["CumWeight"] = df[weight_col].cumsum()
    cutoff = total_weight * top_pct
    df_top = df[df["CumWeight"] <= cutoff]

    total_weighted_value = (df[value_col] * df[weight_col]).sum()
    top_weighted_value = (df_top[value_col] * df_top[weight_col]).sum()

    return top_weighted_value / total_weighted_value if total_weighted_value > 0 else 0.0

def inject_calibrated_pareto_tail(df, top1_share_target=0.20, top10_share_target=0.50, alpha=2.5, seed=42):
    rng = np.random.default_rng(seed)
    df = df.sort_values("Total_Assets", ascending=False).reset_index(drop=True)
    n = len(df)
    top1_n = int(n * 0.01)
    top10_n = int(n * 0.10)

    top1_indices = np.arange(top1_n)
    top10_indices = np.arange(top1_n, top10_n)
    all_top10 = np.arange(top10_n)

    total_net_wealth = df["Net_Wealth"].sum()

    weight_arr = df["Weight"].values
    ta_arr = df["Total_Assets"].values
    dr_arr = df["Debt_Ratio"].values
    debt_arr = df["Debts"].values
    nw_arr = df["Net_Wealth"].values

    # --- Top 1% ---
    top1_weights = weight_arr[top1_indices]
    top1_sample = rng.pareto(alpha, size=top1_n)
    scale1 = (top1_share_target * total_net_wealth) / (top1_sample * top1_weights).sum()
    ta_arr[top1_indices] = top1_sample * scale1

    # --- Next 9% ---
    top10_weights = weight_arr[top10_indices]
    top10_sample = rng.pareto(alpha, size=len(top10_indices))
    target_top10 = top10_share_target * total_net_wealth - top1_share_target * total_net_wealth
    scale10 = target_top10 / (top10_sample * top10_weights).sum()
    ta_arr[top10_indices] = top10_sample * scale10

    # --- Recalculate debts & net wealth ---
    debt_arr[all_top10] = ta_arr[all_top10] * dr_arr[all_top10]
    nw_arr[all_top10] = ta_arr[all_top10] - debt_arr[all_top10]

    df["Total_Assets"] = ta_arr
    df["Debts"] = debt_arr
    df["Net_Wealth"] = nw_arr

    return df

def expand_households_to_individuals(df, base_threshold=600_000, split_prob=0.7, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    df = df.copy()

    if "Original_ID" not in df.columns:
        df["Original_ID"] = df.index
    if "Household_Size" not in df.columns:
        df["Household_Size"] = 1
    if "Weight" not in df.columns:
        df["Weight"] = 1.0

    df["Split"] = (df["Net_Wealth"] > base_threshold) & (rng.random(len(df)) < split_prob)
    df["n_units"] = np.where(df["Split"], 2, 1)

    repeated_df = df.loc[df.index.repeat(df["n_units"])].copy()
    repeated_df["Unit_Index"] = repeated_df.groupby("Original_ID").cumcount()

    ratios = []
    for units in df["n_units"]:
        if units == 2:
            if rng.random() < 0.5:
                ratios.extend([0.8, 0.2])
            else:
                ratios.extend([0.9, 0.1])
        else:
            ratios.append(1.0)

    # Now assign it — this must match exactly in length
    ratios = np.array(ratios)
    if len(ratios) != len(repeated_df):
        raise ValueError("Mismatch between generated ratios and repeated_df length!")

    repeated_df["Ratio"] = ratios

    monetary_cols = [
        "Total_Assets", "Debts", "Real_Assets", "Financial_Assets",
        "Business_Assets", "Income", "Net_Wealth"
    ]
    for col in monetary_cols:
        if col in repeated_df.columns:
            repeated_df[col] *= repeated_df["Ratio"]

    repeated_df["Weight"] *= repeated_df["Ratio"]
    repeated_df["Tax_Unit_ID"] = repeated_df["Unit_Index"] + 1
    repeated_df["Split_From"] = repeated_df["Original_ID"]

    return repeated_df.drop(columns=["n_units", "Unit_Index", "Ratio", "Split"])


def assign_wealth_ranks_balanced_by_region(regions, rng_seed=42):
    """
    Assign wealth ranks to households such that each region receives a fair share of the wealth distribution.

    Parameters:
        regions (array-like): Array of region names (one per household).
        rng_seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of wealth ranks in [0, 1], same length as `regions`.
    """
    regions = np.array(regions)
    unique_regions = np.unique(regions)
    ranks = np.empty(len(regions))

    rng = np.random.default_rng(rng_seed)

    for region in unique_regions:
        mask = (regions == region)
        n = mask.sum()
        region_ranks = np.linspace(0.0001, 1.0, n)
        rng.shuffle(region_ranks)  # Shuffle within region
        ranks[mask] = region_ranks

    return ranks

def generate_and_adjust_households(stats_by_group, region_weights, income_data_file,
                                   household_sizes=None, regions=None):
    if regions is None or household_sizes is None:
        raise ValueError("Must provide both 'regions' and 'household_sizes'.")

    # 1. Create base DataFrame
    wealth_ranks = assign_wealth_ranks_balanced_by_region(regions)
    categories = pd.cut(
        wealth_ranks,
        bins=[0, 0.25, 0.50, 0.75, 0.9, 1.0],
        labels=["under 25", "between 25 and 50", "between 50 and 75", "between 75 and 90", "between 90 and 100"],
        include_lowest=True
    ).astype(str)

    df = pd.DataFrame({
        "Region": regions,
        "Wealth_Rank": wealth_ranks,
        "Category": categories,
        "Household_Size": household_sizes
    })

    # 🛠 FORCE each region to have some wealthy households
    rng = np.random.default_rng(42)
    for region in df["Region"].unique():
        idxs = df[df["Region"] == region].sample(n=min(5, len(df[df["Region"] == region])), random_state=42).index
        df.loc[idxs, "Wealth_Rank"] = rng.uniform(0.95, 1.0, size=len(idxs))
        df.loc[idxs, "Category"] = "between 90 and 100"

    # 2. Merge stats + noise
    df = df.merge(stats_by_group, on="Category", how="left")
    df.dropna(subset=["Total_Assets"], inplace=True)
    np.random.seed(42)
    df["Total_Assets"] *= np.random.normal(1.0, 0.05, len(df))

    mid_mask = (df["Wealth_Rank"] > 0.3) & (df["Wealth_Rank"] <= 0.9)
    df.loc[mid_mask, "Total_Assets"] *= np.random.normal(1.0, 0.15, mid_mask.sum())

    bot_mask = df["Wealth_Rank"] <= 0.5
    df.loc[bot_mask, "Total_Assets"] *= np.random.normal(1.2, 0.25, bot_mask.sum())

    # 3. Construct other components
    df["Business_Assets"] = df.get("Business_Assets", 0.0)
    if df["Business_Assets"].isna().all() and "Business_Asset_Ratio" in df.columns:
        df["Business_Assets"] = df["Total_Assets"] * df["Business_Asset_Ratio"]
    df["Business_Assets"] = df["Business_Assets"].fillna(0.0)
    df["Debts"] = df["Total_Assets"] * df["Debt_Ratio"]
    df["Net_Wealth"] = df["Total_Assets"] - df["Debts"]
    df["Net_Wealth"] = df["Net_Wealth"].clip(lower=7000)
    df["Total_Assets"] = df["Net_Wealth"] + df["Debts"]
    df["Real_Assets"] = df["Total_Assets"] * df["Real_Asset_Ratio"]
    df["Financial_Assets"] = df["Total_Assets"] * df["Financial_Asset_Ratio"]

    # 4. Assign income
    income_data = pd.read_csv(income_data_file)
    income_data = income_data[(income_data["element"].str.contains("TOTAL INCOME", case=False)) &
                              (income_data["breakdown"] == "NET WEALTH PERCENTILE") &
                              (income_data["estadistico"].str.upper() == "MEAN") &
                              (income_data["wave"] == 2022)]
    income_map = dict(zip(income_data["category"].str.strip().str.lower(), income_data["value"] * 1000))
    df["Income"] = df["Category"].map(lambda cat: np.random.normal(
        max(1, income_map.get(cat, 0)), 0.05 * max(1, income_map.get(cat, 0))))

    df["Original_ID"] = df.index

    # 5. Expand to individuals
    df_individuals = expand_households_to_individuals(df, base_threshold=1_000_000)

    return df_individuals, df[["Original_ID", "Household_Size"]]

def assign_declarant_weights(df):
    df = df.copy()
    df["Declarant_Weight"] = df.get("Weight", 1.0)
    return df

def get_personal_exemption(region):
    return 500_000 if region in ["catalonia", "extremadura", "valencia"] else 700_000

def compute_total_exemption(row):
    personal_exemption = get_personal_exemption(row["Region"])
    primary_exempt = min(row.get("Adj_Real_Assets", 0), 300_000)
    return personal_exemption + primary_exempt + row.get("Business_Exemption", 0.0)

def assign_erosion(df, max_erosion=0.40, base_dropouts=True, verbose=False):

    df = df.copy()

    # Base erosion by wealth percentile
    base_erosion = np.select(
        [df["Wealth_Rank"] > 0.999, df["Wealth_Rank"] > 0.99, df["Wealth_Rank"] > 0.90, df["Wealth_Rank"] > 0.75],
        [0.35, 0.25, 0.15, 0.07],
        default=0.02
    )

    # Modifiers
    modifier = (
        1.0
        + 0.10 * (df["Business_Asset_Ratio"] > 0.2)
        + 0.02 * (df["Real_Asset_Ratio"] > 0.4)
        + 0.08 * (df["Financial_Asset_Ratio"] > 0.4)
        + 0.05 * (df["Income"] < 0.6 * df["Adj_Net_Wealth"])
    )

    erosion_factor = np.minimum(base_erosion * modifier, max_erosion)
    df["Erosion_Factor"] = erosion_factor

    if base_dropouts:
        nw = df["Adj_Net_Wealth"]
        dropout_prob = np.where(
            nw < 2_000_000, 0.0,
            np.where(
                nw < 10_000_000,
                (nw - 2_000_000) / 8_000_000 * 0.10,
                0.10 + np.minimum((nw - 10_000_000) / 90_000_000 * 0.20, 0.20)
            )
        )
        df["Dropout_Prob"] = dropout_prob
        df["Dropout"] = np.random.binomial(1, dropout_prob)
    else:
        df["Dropout_Prob"] = 0.0
        df["Dropout"] = 0

    if verbose:
        print(f"📉 Avg Erosion Factor: {df['Erosion_Factor'].mean():.3f}")
        print(f"🚪 Dropouts: {df['Dropout'].sum()} / {len(df)}")

    return df


def apply_migration_module(df, thresholds=None, base_probs=None, max_ratio_bump=0.01, verbose=False):

    df = df.copy()

    # Default thresholds and probabilities
    thresholds = thresholds or {"top_01": 0.999, "top_1": 0.99, "top_5": 0.95}
    base_probs = base_probs or {"top_01": 0.010, "top_1": 0.005, "top_5": 0.002}

    def compute_prob(row):
        wealth_rank = row.get("Wealth_Rank", 0)
        prob = 0.0
        if wealth_rank > thresholds["top_01"]:
            prob = base_probs["top_01"]
        elif wealth_rank > thresholds["top_1"]:
            prob = base_probs["top_1"]
        elif wealth_rank > thresholds["top_5"]:
            prob = base_probs["top_5"]

        if "Wealth_Tax_Baseline" in row and "Adj_Net_Wealth" in row:
            ratio = row["Wealth_Tax_Baseline"] / (row["Adj_Net_Wealth"] + 1e-6)
            prob *= 1 + min(ratio, max_ratio_bump)

        return min(prob, 1.0)

    df["Migration_Prob"] = df.apply(compute_prob, axis=1)
    df["Migration_Exit"] = np.random.rand(len(df)) < df["Migration_Prob"]

    df.loc[df["Migration_Exit"], "Taxable_Wealth_Eroded"] = 0.0
    df.loc[df["Migration_Exit"], "Wealth_Tax"] = 0.0

    if verbose:
        print(f"✈️ Migrants: {df['Migration_Exit'].sum()} / {len(df)}")

    return df

def generate_tax_diagnostics(df):
    filtered = df[df["Is_Taxpayer"] == True]
    diagnostics = {
        "Total_Tax_Revenue": (filtered["Wealth_Tax"] * filtered["Final_Weight"]).sum(),
        "Top_1_Wealth_Share": share_in_top_percentile(filtered, "Net_Wealth", "Final_Weight", 0.01),
        "Declarant_Count": filtered.shape[0],
    }
    print("\n Diagnostics:")
    for k, v in diagnostics.items():
        print(f"{k}: {v:,.2f}")
    return diagnostics

def calculate_ip_tax(base, region="default"):
    regional_scales = {
        "Catalonia": [(0, 167129.45, 0.0021), (167129.45, 334252.88, 0.00315), (334252.88, 668499.75, 0.00525),
                      (668499.75, 1336999.5, 0.00945), (1336999.5, 2673999.0, 0.01365), (2673999.0, 5347998.03, 0.01785),
                      (5347998.03, 10695996.06, 0.02205), (10695996.06, 19999999.99, 0.02525), (19999999.99, float("inf"), 0.0348)],
        "Madrid": [(0, 167129.45, 0.002), (167129.45, 334252.88, 0.003), (334252.88, 668499.75, 0.005),
                   (668499.75, 1336999.5, 0.009), (1336999.5, 2673999.0, 0.013), (2673999.0, 5347998.03, 0.017),
                   (5347998.03, 10695996.06, 0.021), (10695996.06, float("inf"), 0.025)],
        "Extremadura": [(0, 167129.45, 0.002), (167129.45, 334252.88, 0.003), (334252.88, 668499.75, 0.005),
                        (668499.75, 1336999.5, 0.009), (1336999.5, 2673999.0, 0.013), (2673999.0, 5347998.03, 0.017),
                        (5347998.03, 10695996.06, 0.021), (10695996.06, float("inf"), 0.0375)],
        "Galicia": [(0, 167129.45, 0.002), (167129.45, 334252.88, 0.003), (334252.88, 668499.75, 0.005),
                    (668499.75, 1336999.5, 0.009), (1336999.5, 2673999.0, 0.013), (2673999.0, 5347998.03, 0.017),
                    (5347998.03, 10695996.06, 0.021), (10695996.06, float("inf"), 0.035)],
        "Asturias": [(0, 167129.45, 0.002), (167129.45, 334252.88, 0.003), (334252.88, 668499.75, 0.005),
                     (668499.75, 1336999.5, 0.009), (1336999.5, 2673999.0, 0.013), (2673999.0, 5347998.03, 0.017),
                     (5347998.03, 10695996.06, 0.021), (10695996.06, float("inf"), 0.025)],
        "Valencia": [(0, 167129.45, 0.0025), (167129.45, 334252.88, 0.0035), (334252.88, 668499.75, 0.0055),
                     (668499.75, 1336999.5, 0.0095), (1336999.5, 2673999.0, 0.0135), (2673999.0, 5347998.03, 0.0175),
                     (5347998.03, 10695996.06, 0.0215), (10695996.06, float("inf"), 0.035)],
        "default": [(0, 167129.45, 0.002), (167129.45, 334252.88, 0.003), (334252.88, 668499.75, 0.005),
                    (668499.75, 1336999.51, 0.009), (1336999.51, 2673999.01, 0.013), (2673999.01, 5347998.03, 0.017),
                    (5347998.03, 10695996.06, 0.021), (10695996.06, float("inf"), 0.025)]
    }

    brackets = regional_scales.get(region, regional_scales["default"])
    tax = 0.0

    for lower, upper, rate in brackets:
        if base > lower:
            taxable = min(base, upper) - lower
            tax += taxable * rate
        else:
            break

    return tax


def simulate_pit(incomes):
    brackets = np.array([12450, 20200, 35200, 60000, 300000, np.inf])
    rates = np.array([0.19, 0.24, 0.30, 0.37, 0.45, 0.47])

    taxes = np.zeros_like(incomes, dtype=np.float64)
    last_limit = 0.0

    for limit, rate in zip(brackets, rates):
        taxable = np.minimum(incomes, limit) - last_limit
        taxable = np.maximum(taxable, 0)
        taxes += taxable * rate
        last_limit = limit

    return taxes

def apply_tax_cap_and_adjustments(df):
    df = df.copy()
    df["Cap"] = 0.60 * df["Income"]
    over_limit = df["Wealth_Tax"] + df["PIT_Liability"] > df["Cap"]
    df.loc[over_limit, "Wealth_Tax"] = np.maximum(
        0.2 * df.loc[over_limit, "Wealth_Tax"],
        df.loc[over_limit, "Cap"] - df.loc[over_limit, "PIT_Liability"]
    )
    df["Weighted_Wealth_Tax"] = df["Wealth_Tax"] * df["Declarant_Weight"]
    return df

def run_tax_simulation(df, biz_prob=1):
    df = df.copy()

    df["Region_Scale"] = df["Region"].map(region_scaling).fillna(1.0)
    if "Total_Assets_Pareto" in df.columns:
        df["Total_Assets"] = df["Total_Assets_Pareto"]

    df["Debts"] = df["Total_Assets"] * df["Debt_Ratio"]
    df["Net_Wealth"] = df["Total_Assets"] - df["Debts"]
    df["Real_Assets"] = df["Total_Assets"] * df["Real_Asset_Ratio"]
    df["Financial_Assets"] = df["Total_Assets"] * df["Financial_Asset_Ratio"]
    df["Business_Assets"] = df["Total_Assets"] * df["Business_Asset_Ratio"]

    df["Adj_Real_Assets"] = df["Real_Assets"] * 0.75
    df["Adj_Financial_Assets"] = df["Financial_Assets"]
    df["Adj_Business_Assets"] = df["Business_Assets"] * 0.70
    df["Adj_Total_Assets"] = df["Adj_Real_Assets"] + df["Adj_Financial_Assets"] + df["Adj_Business_Assets"]
    df["Adj_Net_Wealth"] = df["Adj_Total_Assets"] - df["Debts"]

    df["Business_Exemption"] = 0.0
    eligible = (
        (df["Business_Asset_Ratio"] > 0.2) & (df["Income"] > 30_000) & (np.random.rand(len(df)) < biz_prob)
    )
    df["Primary_Residence_Exempt"] = df["Adj_Real_Assets"].clip(upper=300_000)
    print(df["Business_Asset_Ratio"].describe())
    print(f"Eligible for exemption: {(eligible.sum())} / {len(df)}")

    exemption_map = {
        "catalonia": 500_000, "extremadura": 500_000, "valencia": 500_000,
        "asturias": 700_000, "galicia": 700_000, "madrid": 700_000
    }
    df["Personal_Exemption"] = df["Region"].map(exemption_map).fillna(700_000)

    reclass_mask = df["Business_Asset_Ratio"] > 0.2
    df["Business_Reclass"] = 0.0
    df.loc[reclass_mask, "Business_Reclass"] = df.loc[reclass_mask, "Adj_Business_Assets"] * 0.2
    df["Adj_Business_Assets"] -= df["Business_Reclass"]
    df["Adj_Total_Assets"] = df["Adj_Real_Assets"] + df["Adj_Financial_Assets"] + df["Adj_Business_Assets"]
    df["Adj_Net_Wealth"] = df["Adj_Total_Assets"] - df["Debts"]

    df["Gross_Tax_Base"] = df["Adj_Net_Wealth"] - df["Primary_Residence_Exempt"] - df["Business_Exemption"]
    df["Net_Tax_Base"] = (df["Gross_Tax_Base"] - df["Personal_Exemption"]).clip(lower=0)
    df["Gross_Assets"] = df["Adj_Total_Assets"]
    df["Is_Declarant"] = (df["Net_Tax_Base"] > 0) | (df["Gross_Assets"] > 2_000_000)
    df["Is_Taxpayer"] = df["Is_Declarant"]

    df = assign_erosion(df)

    df["Liquid_Heavy"] = df["Financial_Asset_Ratio"] > 0.4
    df.loc[df["Liquid_Heavy"], "Income"] *= 0.95
    df["Business_High"] = df["Business_Asset_Ratio"] > 0.2
    df["Business_Low"] = ~df["Business_High"]
    df.loc[df["Business_High"], "Erosion_Factor"] *= 1.05
    df.loc[df["Business_Low"], "Erosion_Factor"] *= 1.10
    df["Erosion_Factor"] = df["Erosion_Factor"].clip(upper=0.3)

    df["Taxable_Wealth"] = (
        df["Adj_Net_Wealth"] - df["Personal_Exemption"] - df["Primary_Residence_Exempt"] - df["Business_Exemption"]
    ).clip(lower=0)
    df["Taxable_Wealth_Baseline"] = df["Taxable_Wealth"]
    df["Taxable_Wealth_Eroded"] = df["Taxable_Wealth"] * (1 - df["Erosion_Factor"])

    df["Wealth_Tax"] = df.apply(lambda row: calculate_ip_tax(row["Taxable_Wealth_Eroded"], row["Region"]), axis=1)
    df.loc[df["Dropout"] > 0, "Wealth_Tax"] = 0

    df["PIT_Liability"] = simulate_pit(df["Income"].values)
    df["Cap"] = 0.60 * df["Income"]
    over_limit = df["Wealth_Tax"] + df["PIT_Liability"] > df["Cap"]
    df.loc[over_limit, "Wealth_Tax"] = np.maximum(
        0.2 * df.loc[over_limit, "Wealth_Tax"],
        df.loc[over_limit, "Cap"] - df.loc[over_limit, "PIT_Liability"]
    )

    return df

def apply_baseline_behavioral_erosion(df: pd.DataFrame, verbose=False):

    df = df.copy()
    assert "Erosion_Factor" in df.columns, "Erosion_Factor must be computed before applying behavioral erosion"

    # Apply erosion
    df["Taxable_Wealth_Baseline_Eroded"] = df["Taxable_Wealth_Baseline"] * (1 - df["Erosion_Factor"])
    df["Wealth_Tax_Baseline_Eroded"] = df.apply(
        lambda row: calculate_ip_tax(row["Taxable_Wealth_Baseline_Eroded"], row["Region"]), axis=1
    )
    df["Weighted_Wealth_Tax_Baseline_Eroded"] = (
        df["Wealth_Tax_Baseline_Eroded"] * df["Final_Weight"]
    )

    # Compute summary
    baseline_total = (df["Wealth_Tax_Baseline"] * df["Final_Weight"]).sum() if "Wealth_Tax_Baseline" in df.columns else None
    eroded_total = df["Weighted_Wealth_Tax_Baseline_Eroded"].sum()
    revenue_gap = baseline_total - eroded_total if baseline_total is not None else None
    gap_pct = (revenue_gap / baseline_total) * 100 if baseline_total else None

    summary = {
        "Baseline_Wealth_Tax": baseline_total,
        "Eroded_Wealth_Tax": eroded_total,
        "Revenue_Gap": revenue_gap,
        "Revenue_Gap_%": gap_pct
    }

    if verbose:
        print("\n📉 Revenue Gap Due to Behavioral Erosion:")
        for k, v in summary.items():
            print(f"{k}: {v:,.2f}" if v is not None else f"{k}: N/A")

    return df, summary

region_scaling = {
    "asturias": 0.7,
    "catalonia": 1.7,
    "extremadura": 0.7,
    "galicia": 1.5,
    "valencia": 0.9,
    "madrid": 0.2
}

def apply_region_multipliers(df, multipliers, recompute=True):

    df = df.copy()
    all_regions = df["Region"].unique()

    # Validation for required ratio columns
    required_ratios = ["Debt_Ratio", "Real_Asset_Ratio", "Financial_Asset_Ratio"]
    for col in required_ratios:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NaN values")

    for region in all_regions:
        if region not in multipliers:
            print(f" Warning: No scaling multiplier found for region '{region}'. Using default factor 1.0.")
        factor = multipliers.get(region, 1.0)
        mask = df["Region"] == region

        # Scale total assets
        df.loc[mask, "Total_Assets"] *= factor

        if recompute:
            # Recompute dependent fields
            df.loc[mask, "Debts"] = df.loc[mask, "Total_Assets"] * df.loc[mask, "Debt_Ratio"]
            df.loc[mask, "Net_Wealth"] = df.loc[mask, "Total_Assets"] - df.loc[mask, "Debts"]
            df.loc[mask, "Real_Assets"] = df.loc[mask, "Total_Assets"] * df.loc[mask, "Real_Asset_Ratio"]
            df.loc[mask, "Financial_Assets"] = df.loc[mask, "Total_Assets"] * df.loc[mask, "Financial_Asset_Ratio"]

            if "Business_Asset_Ratio" in df.columns:
                if df.loc[mask, "Business_Asset_Ratio"].isna().any():
                    print(f"⚠️ Warning: NaN values in 'Business_Asset_Ratio' for region '{region}'")
                df.loc[mask, "Business_Assets"] = df.loc[mask, "Total_Assets"] * df.loc[mask, "Business_Asset_Ratio"]
            else:
                print(f"ℹ️ Info: 'Business_Asset_Ratio' not found. Setting Business_Assets to 0 for region '{region}'")
                df.loc[mask, "Business_Assets"] = 0.0

    df.reset_index(drop=True, inplace=True)
    print("\n✅ Region multipliers applied. Preview of updated DataFrame:")
    print(df.head())
    return df

def scale_final_weights_by_taxpayer_counts(df, region_targets_quota):
    df = df.copy()
    df["Region"] = df["Region"].str.strip().str.lower()

    df["Scaled_Taxpayer_Weight"] = df["Final_Weight"]  # start by preserving existing weight

    for region, target_count in region_targets_quota.items():
        mask = (df["Region"] == region) & (df["Is_Taxpayer"] == 1)
        regional_declarants = df[mask]

        if regional_declarants.empty:
            print(f" No taxpayers found in region '{region}', skipping.")
            continue

        total_weight = regional_declarants["Final_Weight"].sum()
        if total_weight == 0:
            print(f" Zero simulated taxpayer weight in region '{region}', skipping.")
            continue

        scaling_factor = target_count / total_weight
        df.loc[mask, "Scaled_Taxpayer_Weight"] *= scaling_factor

    # Set Final_Weight to this adjusted version
    df["Final_Weight"] = df["Scaled_Taxpayer_Weight"]
    df.drop(columns=["Scaled_Taxpayer_Weight"], inplace=True)

    return df

def compare_to_observed(households_df: pd.DataFrame) -> pd.DataFrame:
    # Load and clean observed data
    observed_df = pd.read_csv("Cleaned_Regional_Wealth_Tax_Data.csv")
    observed_df["Region"] = observed_df["Region"].str.strip().str.lower()

    # Filter only the "resultado de la declaración" rows
    observed_clean = observed_df[
        observed_df["Variable"].str.strip().str.lower() == "resultado de la declaración"
    ].copy()

    # Parse numeric values in Spanish format
    observed_clean["Total_Revenue"] = (
        observed_clean["Importe"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    observed_clean["Total_Revenue"] = pd.to_numeric(observed_clean["Total_Revenue"], errors="coerce")
    observed_clean = observed_clean[["Region", "Total_Revenue"]]

    # Prepare simulated data
    df = households_df.copy()
    df["Region"] = df["Region"].str.strip().str.lower()
    df_taxpayers = df[df["Is_Taxpayer"] == True].copy()
    df_taxpayers["Weighted_Wealth_Tax"] = df_taxpayers["Wealth_Tax"] * df_taxpayers["Final_Weight"]

    simulated_revenue = (
        df_taxpayers.groupby("Region", as_index=False)["Weighted_Wealth_Tax"]
        .sum()
        .rename(columns={"Weighted_Wealth_Tax": "Simulated_Actual_Revenue"})
    )

    # Merge and calculate % gap (handles missing observed values)
    merged = pd.merge(simulated_revenue, observed_clean, on="Region", how="outer")
    merged.fillna({"Simulated_Actual_Revenue": 0.0, "Total_Revenue": 0.0}, inplace=True)

    merged["Gap_%"] = np.where(
        merged["Total_Revenue"] > 0,
        100 * (merged["Simulated_Actual_Revenue"] - merged["Total_Revenue"]) / merged["Total_Revenue"],
        np.nan
    )

    # Format for human-readable output
    merged["Simulated_Actual_Revenue"] = merged["Simulated_Actual_Revenue"].map("${:,.0f}".format)
    merged["Total_Revenue"] = merged["Total_Revenue"].map("${:,.0f}".format)
    merged["Gap_%"] = merged["Gap_%"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")

    # Add total row
    total_sim = df_taxpayers["Weighted_Wealth_Tax"].sum()
    total_obs = observed_clean["Total_Revenue"].sum()
    total_gap_pct = (
        100 * (total_sim - total_obs) / total_obs if total_obs > 0 else np.nan
    )
    total_row = pd.DataFrame([{
        "Region": "TOTAL",
        "Simulated_Actual_Revenue": "${:,.0f}".format(total_sim),
        "Total_Revenue": "${:,.0f}".format(total_obs),
        "Gap_%": f"{total_gap_pct:.2f}%" if pd.notna(total_gap_pct) else "N/A"
    }])

    result = pd.concat([merged, total_row], ignore_index=True)

    # Print report
    print("\n📊 Revenue Comparison with Observed:")
    print(result.to_string(index=False))
    return result
def generate_region_summary(df_region, region_name):
    df_region = df_region.copy()

    # Full base: no exemptions
    df_region["FullBase_Taxable"] = df_region["Adj_Net_Wealth"]
    df_region["Wealth_Tax_FullBase"] = df_region.apply(
        lambda row: calculate_ip_tax(row["FullBase_Taxable"], row["Region"]), axis=1
    )

    # Actual weighted tax
    df_region["Weighted_Wealth_Tax"] = df_region["Wealth_Tax"] * df_region["Final_Weight"]
    df_region["Weighted_Wealth_Tax_FullBase"] = df_region["Wealth_Tax_FullBase"] * df_region["Final_Weight"]

    total_revenue = df_region["Weighted_Wealth_Tax"].sum()
    fullbase_revenue = df_region["Weighted_Wealth_Tax_FullBase"].sum()
    exemption_gap = fullbase_revenue - total_revenue
    exemption_gap_pct = 100 * exemption_gap / fullbase_revenue if fullbase_revenue > 0 else np.nan

    # Decentralization: apply national rule (Asturias) to all
    df_region["Wealth_Tax_BaselineRule"] = df_region["Adj_Net_Wealth"].apply(
        lambda w: calculate_ip_tax(w, region="Asturias")
    )
    df_region["Weighted_Wealth_Tax_BaselineRule"] = df_region["Wealth_Tax_BaselineRule"] * df_region["Final_Weight"]

    baseline_rule_revenue = df_region["Weighted_Wealth_Tax_BaselineRule"].sum()
    decentralization_gap = baseline_rule_revenue - total_revenue
    decentralization_gap_pct = 100 * decentralization_gap / total_revenue if total_revenue > 0 else np.nan

    return {
        "Region": region_name,
        "Revenue_With_Exemptions": total_revenue,
        "Revenue_No_Exemptions": fullbase_revenue,
        "Exemption_Gap": exemption_gap,
        "Exemption_Gap_%": exemption_gap_pct,
        "BaselineRule_Revenue": baseline_rule_revenue,
        "Decentralization_Gap": decentralization_gap,
        "Decentralization_Gap_%": decentralization_gap_pct,
        "Num_Taxpayers": df_region["Is_Taxpayer"].sum(),
        "Avg_Taxpayer_ETR": (
            df_region.loc[df_region["Is_Taxpayer"], "Wealth_Tax"].sum()
            / df_region.loc[df_region["Is_Taxpayer"], "Adj_Net_Wealth"].sum()
        ) if df_region["Is_Taxpayer"].any() else np.nan,
        "Avg_Top10_ETR": (
            df_region.loc[df_region["Wealth_Rank"] > 0.9, "Wealth_Tax"].sum()
            / df_region.loc[df_region["Wealth_Rank"] > 0.9, "Adj_Net_Wealth"].sum()
        ) if (df_region["Wealth_Rank"] > 0.9).any() else np.nan,
    }


import matplotlib.pyplot as plt

def compute_global_inequality_stats(df, weight_col="Final_Weight", plot=True):
    df = df.copy()
    df = df[(df["Adj_Net_Wealth"] >= 0) & (df[weight_col] > 0)]
    df["Post_Tax_Wealth"] = df["Adj_Net_Wealth"] - df["Wealth_Tax"]

    def weighted_gini(x, w):
        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        w = w[sorted_idx]
        cumw = np.cumsum(w)
        cumxw = np.cumsum(x * w)
        return 1 - 2 * np.sum(w * (cumxw - x * w / 2)) / (cumw[-1] * cumxw[-1])

    def share(df, col, pct):
        df = df[df[col] > 0].sort_values(col, ascending=False).copy()
        df["cum_w"] = df[weight_col].cumsum()
        cutoff = df[weight_col].sum() * pct
        top = df[df["cum_w"] <= cutoff]
        return (top[col] * top[weight_col]).sum() / (df[col] * df[weight_col]).sum()

    gini_pre = weighted_gini(df["Adj_Net_Wealth"].values, df[weight_col].values)
    gini_post = weighted_gini(df["Post_Tax_Wealth"].values, df[weight_col].values)

    shares = {
        "Top 1% (Pre-Tax)": share(df, "Adj_Net_Wealth", 0.01),
        "Top 1% (Post-Tax)": share(df, "Post_Tax_Wealth", 0.01),
        "Top 10% (Pre-Tax)": share(df, "Adj_Net_Wealth", 0.10),
        "Top 10% (Post-Tax)": share(df, "Post_Tax_Wealth", 0.10),
    }

    # Lorenz Curve
    if plot:
        df_sorted = df.sort_values("Adj_Net_Wealth")
        cumw = np.cumsum(df_sorted[weight_col]) / df_sorted[weight_col].sum()
        cumx = np.cumsum(df_sorted["Adj_Net_Wealth"] * df_sorted[weight_col]) / (df_sorted["Adj_Net_Wealth"] * df_sorted[weight_col]).sum()
        plt.figure(figsize=(6, 6))
        plt.plot(np.insert(cumw, 0, 0), np.insert(cumx, 0, 0), label="Pre-Tax Lorenz")
        plt.plot([0, 1], [0, 1], '--', color='gray', label="Equality")
        plt.title("Lorenz Curve (Pre-Tax Wealth)")
        plt.xlabel("Cumulative Population Share")
        plt.ylabel("Cumulative Wealth Share")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    print(f"\n📊 Global Wealth Inequality Metrics:")
    print(f"  Gini (Pre-Tax): {gini_pre:.4f}")
    print(f"  Gini (Post-Tax): {gini_post:.4f}")
    for k, v in shares.items():
        print(f"  {k} Wealth Share: {v:.2%}")

    return {
        "Gini_Pre": gini_pre,
        "Gini_Post": gini_post,
        **shares
    }

def main():
    summary_rows = []
    try:
        POP_FILE = "Regional_Age_Bin_Population_Shares.csv"
        INCOME_FILE = "eff_incomedata.csv"
        STATS_FILE = "eff_data.xlsx"


        # === LOAD DATA ===
        base_population = calculate_population_over_30(POP_FILE)
        eff_df = load_eff_data(STATS_FILE)
        _, group_stats_df = process_eff_assets_income(eff_df)
        revenue_df, region_weights = load_population_and_revenue_data(POP_FILE)

        # Normalize wealth groups
        valid_wealth_groups = [
            "under 25",
            "between 25 and 50",
            "between 50 and 75",
            "between 75 and 90",
            "between 90 and 100"
        ]
        group_stats_df["Category"] = group_stats_df["Category"].str.lower().str.strip()
        group_stats_df = group_stats_df[group_stats_df["Category"].isin(valid_wealth_groups)]

        # === SYNTHETIC HOUSEHOLD GENERATION ===
        simulated_population = int(150_000 * 1.6)
        household_meta = generate_scaled_households(region_weights, simulated_population=simulated_population)
        regions = household_meta["Region"].values
        household_sizes = household_meta["Household_Size"].values

        individuals, household_sizes_lookup = generate_and_adjust_households(
            group_stats_df, region_weights, INCOME_FILE,
            household_sizes=household_sizes, regions=regions
        )

        assert "Household_Size" in individuals.columns, "❌ Household_Size missing."
        individuals["Final_Weight"] = 1.0
        individuals = inject_calibrated_pareto_tail(
            individuals,
            top1_share_target=0.20,
            alpha=2.5,
            seed=42
        )

        individuals["Total_Assets_Pareto"] = individuals["Total_Assets"]

        df_sorted = individuals.sort_values("Total_Assets", ascending=False)
        cum_weights = df_sorted["Final_Weight"].cumsum()
        total_weight = cum_weights.iloc[-1]

        top_1_mask = cum_weights <= 0.01 * total_weight
        top_10_mask = cum_weights <= 0.10 * total_weight

        top_1_share = (df_sorted.loc[top_1_mask, "Total_Assets"] * df_sorted.loc[top_1_mask, "Final_Weight"]).sum() / \
                      (df_sorted["Total_Assets"] * df_sorted["Final_Weight"]).sum()

        top_10_share = (df_sorted.loc[top_10_mask, "Total_Assets"] * df_sorted.loc[top_10_mask, "Final_Weight"]).sum() / \
                       (df_sorted["Total_Assets"] * df_sorted["Final_Weight"]).sum()

        print(f"Top 1% share after Pareto: {top_1_share:.2%}")
        print(f"Top 10% share after Pareto: {top_10_share:.2%}")

        # === Tax Simulation ===
        taxed_individuals = individuals.copy()
        taxed_individuals["Taxable_Wealth_FullBase"] = taxed_individuals["Net_Wealth"]
        taxed_individuals = run_tax_simulation(individuals)
        taxed_individuals.loc[taxed_individuals["Region"].str.lower() == "madrid", "Wealth_Tax"] = 0.0


        taxed_individuals = assign_erosion(taxed_individuals, max_erosion=0.30, verbose=True)
        taxed_individuals = apply_migration_module(
            taxed_individuals,
            thresholds={"top_01": 0.999, "top_1": 0.99, "top_5": 0.95},
            base_probs={"top_01": 0.015, "top_1": 0.007, "top_5": 0.002},
            max_ratio_bump=0.015,
            verbose=True
        )
        taxed_individuals["Taxable_Wealth_Baseline"] = taxed_individuals["Net_Tax_Base"]
        taxed_individuals["Wealth_Tax_Baseline"] = taxed_individuals.apply(
            lambda row: calculate_ip_tax(row["Taxable_Wealth_Baseline"], row["Region"]), axis=1
        )

        taxed_individuals, erosion_summary = apply_baseline_behavioral_erosion(taxed_individuals, verbose=True)

        summary_rows = []

        for region in taxed_individuals["Region"].unique():
            region_df = taxed_individuals[taxed_individuals["Region"] == region].copy()

            # Recalculate tax under Asturias rule
            df_asturias = region_df.copy()
            df_asturias["Wealth_Tax"] = df_asturias.apply(
                lambda row: calculate_ip_tax(row["Taxable_Wealth_Baseline"], "Asturias"), axis=1
            )

            # Compute summary row
            summary_row = generate_region_summary(
                df_region=region_df,
                region_name=region,
            )
            summary_rows.append(summary_row)

        generate_tax_diagnostics(taxed_individuals)

        inequality_summary = compute_global_inequality_stats(taxed_individuals, weight_col="Final_Weight", plot=True)
        print(inequality_summary)
        taxed_individuals.to_csv("simulated_thesis.csv", index=False)

        compare_to_observed(taxed_individuals)

    except Exception as e:
        logging.exception(f"Pipeline execution failed: {e}")
        print("⚠️ Simulation failed due to error above.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("region_policy_comparison_summary.csv", index=False)
    print("📊 Exported region comparison summary to region_policy_comparison_summary.csv")


if __name__ == "__main__":
    main()
