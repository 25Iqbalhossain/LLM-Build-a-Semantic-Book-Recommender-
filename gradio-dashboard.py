import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

import gradio as gr

# Just in case: disable TF backend for transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

load_dotenv()

# ---------- Load books data ----------
# isbn13 শুরু থেকেই string হিসেবে পড়ি
books = pd.read_csv("books_with_emotions.csv", dtype={"isbn13": str})

# isbn13 থেকে শুধু 10–13 digit রাখি (".0" বা অন্য কিছু থাকলে কেটে যাবে)
books["isbn13"] = (
    books["isbn13"]
    .astype(str)
    .str.extract(r"(\d{10,13})", expand=False)
    .str.strip()
)

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ---------- Ensure simple_categories exists ----------
if "simple_categories" not in books.columns:
    if "categories" in books.columns:
        def simplify_category(cat):
            if pd.isna(cat):
                return "Other"
            c = str(cat).lower()

            # basic heuristics – চাইলে পরে কাস্টমাইজ করতে পারো
            if "young adult" in c or "ya " in c:
                return "Young Adult"
            if "children" in c or "kids" in c or "juvenile" in c:
                return "Children"
            if "fiction" in c:
                return "Fiction"
            if any(x in c for x in ["business", "management", "finance", "economics"]):
                return "Business"
            if any(x in c for x in ["self-help", "self help", "personal development"]):
                return "Self-Help"
            return "Non-Fiction"

        books["simple_categories"] = books["categories"].apply(simplify_category)
    else:
        # একদমই কোনো ক্যাটাগরি না থাকলে fallback
        books["simple_categories"] = "Unknown"

# ---------- Build vector store with Hugging Face embeddings ----------
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

raw_documents = TextLoader(
    "tagged_description.txt",
    encoding="utf-8",
    autodetect_encoding=True
).load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=0
)
documents = text_splitter.split_documents(raw_documents)

db_books = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="books",
    persist_directory="db_books",
)


# ---------- Recommendation logic ----------
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    isbn_list: list[str] = []
    for rec in recs:
        text = rec.page_content.strip('"').strip()
        # find first 10–13 digit sequence in the text
        m = re.search(r"(\d{10,13})", text)
        if m:
            isbn13 = m.group(1)
            isbn_list.append(isbn13)
            # যদি 13-digit হয়, তাহলে last 10 digit দিয়েও চেষ্টা করব
            if len(isbn13) > 10:
                isbn_list.append(isbn13[-10:])

    if not isbn_list:
        print("No ISBNs found from vector search")
        return books.iloc[0:0]

    # de-duplicate while preserving order
    isbn_list = list(dict.fromkeys(isbn_list))
    print("Found ISBNs:", isbn_list[:10])

    book_recs = books[books["isbn13"].isin(isbn_list)].head(initial_top_k)
    print("Matched books:", len(book_recs))

    if category != "All":
        book_recs = book_recs[
            book_recs["simple_categories"] == category
        ].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row.get("description", "") or ""
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_raw = row.get("authors", "") or ""
        authors_split = [a.strip() for a in authors_raw.split(";") if a.strip()]

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    # IMPORTANT: Gallery-তে None-type ইমেজ পাঠাব না
    # যদি কিছুই না থাকে, empty list ফিরিয়ে দিই
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# ---------- Gradio UI ----------
with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic book recommender (Hugging Face embeddings)")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All",
        )
        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All",
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )


if __name__ == "__main__":
    dashboard.launch()
