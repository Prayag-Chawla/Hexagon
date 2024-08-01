import streamlit as st
import requests
from bs4 import BeautifulSoup
import sqlite3
import os
from PIL import Image
from io import BytesIO
from transformers import pipeline
import pandas as pd
import re
from functools import lru_cache
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define paths for CSV and labels
csv_path = "D:\\Prayag Files\\TIET\\Extras\\Internship\\Ongoing\\hexagon\\Solution\\craters\\labels.csv"
labels_folder = "D:\\Prayag Files\\TIET\\Extras\\Internship\\Ongoing\\hexagon\\Solution\\craters\\train\\labels"

# Initialize the NLP models
qa_pipeline = pipeline("question-answering")

# Database setup
def setup_database():
    conn = sqlite3.connect('dataset.db')
    c = conn.cursor()
    
    # Create or modify tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            url TEXT,
            content TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            file_path TEXT,
            image BLOB,
            num_craters INTEGER,
            coordinates TEXT
        )
    ''')
    
    # Alter the table if necessary to add the coordinates column
    try:
        c.execute('ALTER TABLE images ADD COLUMN coordinates TEXT')
    except sqlite3.OperationalError:
        # The column already exists
        pass
    
    conn.commit()
    return conn, c

conn, c = setup_database()

# Fetch and save article text from a given URL
def fetch_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = " ".join([p.get_text() for p in soup.find_all('p')])
        save_article_to_db(url, article_text)
        return article_text
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching article: {e}")
        return None

# Save article to the database
def save_article_to_db(url, content):
    c.execute("INSERT INTO articles (url, content) VALUES (?, ?)", (url, content))
    conn.commit()

# Fetch and save local image dataset
def fetch_local_images(directory):
    crater_info = pd.read_csv(csv_path)
    images_added = 0

    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for path in image_paths:
        try:
            image_name = os.path.basename(path)
            crater_data = crater_info[crater_info['image_name'].str.lower() == image_name.lower()]
            if not crater_data.empty:
                num_craters = crater_data.iloc[0]['num_craters']
                
                # Load coordinates from the corresponding label file
                label_file = os.path.join(labels_folder, image_name.replace(".jpg", ".txt"))
                if os.path.exists(label_file):
                    with open(label_file, 'r') as lf:
                        coordinates = lf.read().strip().replace('\n', ', ')
                else:
                    coordinates = "No coordinates available"

                with open(path, 'rb') as img_file:
                    image_blob = img_file.read()
                    c.execute("INSERT INTO images (file_path, image, num_craters, coordinates) VALUES (?, ?, ?, ?)",
                              (path, image_blob, num_craters, coordinates))
                    conn.commit()
                    images_added += 1
        except Exception as e:
            st.warning(f"Error adding image {path}: {e}")

    st.write(f"Images added to database: {images_added}")
    
    c.execute("SELECT file_path, num_craters FROM images")
    images = c.fetchall()
    st.write("Images in database:")
    for file_path, num_craters in images:
        st.write(f"File Path: {file_path}, Number of Craters: {num_craters}")
    
    return images_added

# Preprocess text and find the best answer
def find_best_answer(question):
    c.execute("SELECT content FROM articles")
    articles = [row[0] for row in c.fetchall()]
    
    combined_text = articles
    
    if not combined_text:
        return "No articles available for answering."
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined_text)
    
    question_vector = vectorizer.transform([question])
    
    similarities = cosine_similarity(question_vector, vectors)
    
    best_match_idx = np.argmax(similarities)
    best_match_text = combined_text[best_match_idx]
    
    result = qa_pipeline(question=question, context=best_match_text)
    return result['answer']

# Find and display images based on a query
def find_image(query):
    num_craters = None
    words = re.findall(r'\b\d+\b', query)
    if words:
        num_craters = int(words[0])
    
    st.write(f"Query: {query}")
    st.write(f"Extracted number of craters: {num_craters}")

    if num_craters is not None:
        c.execute("SELECT file_path, image, coordinates FROM images WHERE num_craters = ?", (num_craters,))
    else:
        c.execute("SELECT file_path, image, coordinates FROM images")
    
    image_data = c.fetchall()

    if not image_data:
        return None
    
    displayed_images = set()
    images_found = []
    for file_path, image_blob, coordinates in image_data:
        if file_path not in displayed_images:
            try:
                image = Image.open(BytesIO(image_blob))
                images_found.append((image, file_path, coordinates))
                displayed_images.add(file_path)
            except Exception as e:
                st.warning(f"Error opening image {file_path}: {e}")

    return images_found

# Add multiple articles from user input
def add_articles_from_input(article_urls):
    urls = [url.strip() for url in article_urls.split("\n") if url.strip()]
    for url in urls:
        fetch_article_text(url)

# Streamlit UI
def main():
    st.title("Custom Dataset Query Tool")

    st.markdown("""
    <style>
    body {
        background-color: #e0f7fa;
        color: #004d40;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton > button {
        background-color: #00796b;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stTextInput input {
        border: 2px solid #00796b;
        border-radius: 5px;
        padding: 12px;
        font-size: 16px;
    }
    .stTextArea textarea {
        border: 2px solid #00796b;
        border-radius: 5px;
        padding: 12px;
        font-size: 16px;
    }
    .stSelectbox select {
        border: 2px solid #00796b;
        border-radius: 5px;
        padding: 12px;
        font-size: 16px;
    }
    .stHeader {
        color: #004d40;
    }
    @media (max-width: 600px) {
        .stButton > button {
            font-size: 14px;
            padding: 8px 16px;
        }
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            font-size: 14px;
            padding: 8px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a functionality", ["Add Crater Image Dataset", "Add Multiple Articles", "Ask a Question", "Query Images"])

    if page == "Add Crater Image Dataset":
        st.header("Add Crater Image Dataset")
        image_dir = st.text_input("Enter the directory path containing crater images")
        if st.button("Add Image Dataset"):
            if image_dir and os.path.isdir(image_dir):
                fetch_local_images(image_dir)
            else:
                st.warning("Please provide a valid path.")

    elif page == "Add Multiple Articles":
        st.header("Add Multiple Articles")
        article_urls = st.text_area("Enter article URLs (one per line):")
        if st.button("Add Articles"):
            if article_urls:
                add_articles_from_input(article_urls)
                st.success("Articles added to the database.")
            else:
                st.warning("Please enter at least one URL.")

    elif page == "Ask a Question":
        st.header("Ask a Question")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                answer = find_best_answer(question)
                st.write(f"Answer: {answer}")
            else:
                st.warning("Please enter a question.")

    elif page == "Query Images":
        st.header("Query Images")
        query = st.text_input("Enter your query (e.g., 'Give me an image of the moon having 5 craters')")
        if st.button("Find Image"):
            if query:
                images = find_image(query)
                if images:
                    for image, path, coordinates in images:
                        st.image(image, caption=f"Path: {path}\nCoordinates: {coordinates}")
                else:
                    st.write("No images found matching the query.")
            else:
                st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
