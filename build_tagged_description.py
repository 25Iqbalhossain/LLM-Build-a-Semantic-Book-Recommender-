import pandas as pd

books = pd.read_csv("books_with_emotions.csv", dtype={"isbn13": str})

books["isbn13"] = (
    books["isbn13"]
    .astype(str)
    .str.extract(r"(\d{10,13})", expand=False)
    .str.strip()
)

if "simple_categories" not in books.columns:
    books["simple_categories"] = "Unknown"

with open("tagged_description.txt", "w", encoding="utf-8") as f:
    for _, row in books.iterrows():
        isbn = row["isbn13"]
        desc = str(row.get("description", "") or "").replace("\n", " ")
        cat = row.get("simple_categories", "Unknown")
        f.write(f"{isbn} {cat} {desc}\n")

print("tagged_description.txt rebuilt from books_with_emotions.csv")
