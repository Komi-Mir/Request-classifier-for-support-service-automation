import pandas as pd

from preprocess import preprocess_text


INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "correct_dataset.csv"


def main() -> None:
    data = pd.read_csv(INPUT_FILE)

    data["text"] = data["text"].fillna("").astype(str)
    data["text"] = data["text"].apply(preprocess_text)

    data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"Файл обработан и сохранён: {OUTPUT_FILE}")
    print("Размер датасета:", data.shape)
    print("\nРаспределение категорий:")
    print(data["category"].value_counts())


if __name__ == "__main__":
    main()