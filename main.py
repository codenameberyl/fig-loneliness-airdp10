from src.preprocessing import (
    load_fig_loneliness,
    apply_preprocessing,
    validate_dataset
)
from src.eda import(
    run_eda
)

ROOT = "./dataset/"


def main():
    dataset = load_fig_loneliness(ROOT)

    print("\nDataset successfully loaded.")
    print(dataset)

    print("\nColumns in train split:")
    print(dataset["train"].column_names)

    processed_dataset = apply_preprocessing(dataset)

    validate_dataset(processed_dataset, num_samples_to_print=2)

    run_eda(processed_dataset)

if __name__ == "__main__":
    main()
