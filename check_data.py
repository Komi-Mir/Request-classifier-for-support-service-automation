import pandas as pd

df = pd.read_csv("correct_dataset.csv")

print("Размер датасета:", df.shape)

print("\nПропуски:")
print(df.isna().sum())

print("\nРаспределение категорий:")
print(df["category"].value_counts())

print("\nКоличество полных дублей:")
print(df.duplicated().sum())

print("\nКоличество дублей по тексту:")
print(df.duplicated(subset=["text"]).sum())
