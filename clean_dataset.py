import pandas as pd


INPUT_FILE = "correct_dataset.csv"
OUTPUT_FILE = "correct_dataset.csv"

df = pd.read_csv(INPUT_FILE)

print("До очистки:", df.shape)

# Удаляем пропуски в тексте
df["text"] = df["text"].fillna("").astype(str).str.strip()
df = df[df["text"] != ""]

# Удаляем полные дубли
df = df.drop_duplicates()

# Удаляем дубли по тексту
df = df.drop_duplicates(subset=["text"])

df = df.reset_index(drop=True)

print("После очистки:", df.shape)

print("\nПропуски:")
print(df.isna().sum())

print("\nРаспределение категорий:")
print(df["category"].value_counts())

print("\nКоличество полных дублей:")
print(df.duplicated().sum())

print("\nКоличество дублей по тексту:")
print(df.duplicated(subset=["text"]).sum())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nФайл сохранён: {OUTPUT_FILE}")