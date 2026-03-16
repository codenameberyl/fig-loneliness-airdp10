from src.data.loader import load_data, inspect_dataset, dataset_to_dataframe
from src.eda import run_eda
from src.preprocessing import prepare_binary_dataframe, preprocess_for_classical
from src.models import train_and_evaluate_models


def print_stage(title: str):
    print("\n" + "═" * 70)
    print(title)
    print("═" * 70)


def main():

    # ─────────────────────────────────────────
    # STAGE 1: Data Loading
    # ─────────────────────────────────────────
    print_stage("STAGE 1: DATA LOADING")

    dataset = load_data()  # Automatically handles local vs HF
    inspect_dataset(dataset)

    print("\n✅ Data loading complete.")

    # ─────────────────────────────────────────
    # STAGE 2: PREPROCESSING
    # ─────────────────────────────────────────
    print_stage("STAGE 2: PREPROCESSING")

    df = prepare_binary_dataframe(dataset)

    df["clean_text"] = preprocess_for_classical(df["text"])

    print("\nSample cleaned text:")
    print(df["clean_text"].iloc[0][:300])

    print("\n✅ Preprocessing complete.")

    # ─────────────────────────────────────────
    # STAGE 3: EXPLORATORY DATA ANALYSIS
    # ─────────────────────────────────────────
    print_stage("STAGE 3: EXPLORATORY DATA ANALYSIS")

    run_eda(df[["text", "label"]].copy())

    print("\n🎯 EDA completed successfully.")

    print("\nPipeline execution up to EDA finished ✔")

    # ─────────────────────────────────────────
    # STAGE 4: MODEL TRAINING
    # ─────────────────────────────────────────
    print_stage("STAGE 4: MODEL TRAINING")

    results = train_and_evaluate_models(df)

    print("\nModel training and evaluation complete.")
    print(results)  


if __name__ == "__main__":
    main()
