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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import urllib.parse

# Paths for CSV and Labels Folder
CSV_PATH = "D:\\Prayag Files\\TIET\\Extras\\Internship\\Previous\\hexagon\\Solution\\craters\\labels.csv"
LABELS_DIR = "D:\\Prayag Files\\TIET\\Extras\\Internship\\Previous\\hexagon\\Solution\\craters\\train\\labels"

# Initialize the NLP model
qa_pipeline = pipeline("question-answering")

# Database setup
def setup_database():
    conn = sqlite3.connect('dataset.db')
    c = conn.cursor()
    
    # Create or reset tables
    c.execute('''CREATE TABLE IF NOT EXISTS articles
                 (id INTEGER PRIMARY KEY, url TEXT, content TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY, file_path TEXT, image BLOB, num_craters INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS craters
                 (id INTEGER PRIMARY KEY, image_id INTEGER, x_center REAL, y_center REAL, width REAL, height REAL,
                 FOREIGN KEY (image_id) REFERENCES images (id))''')
    
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
        c.execute("INSERT INTO articles (url, content) VALUES (?, ?)", (url, article_text))
        conn.commit()
        return article_text
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching article: {e}")
        return None

# Add image and crater details to the database
def add_image_to_db(image_path, crater_details):
    try:
        with open(image_path, 'rb') as img_file:
            image_blob = img_file.read()
            c.execute("INSERT INTO images (file_path, image, num_craters) VALUES (?, ?, ?)",
                      (image_path, image_blob, len(crater_details)))
            image_id = c.lastrowid
            for crater in crater_details:
                c.execute("INSERT INTO craters (image_id, x_center, y_center, width, height) VALUES (?, ?, ?, ?, ?)",
                          (image_id, crater['x_center'], crater['y_center'], crater['width'], crater['height']))
            conn.commit()
    except Exception as e:
        st.warning(f"Error adding image {image_path}: {e}")

# Fetch and save local image dataset from CSV and label files
def fetch_local_images_from_csv_and_labels(image_dir):
    crater_info = pd.read_csv(CSV_PATH)
    images_added = 0

    try:
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    except Exception as e:
        st.warning(f"Error accessing image directory: {e}")
        return 0

    for path in image_paths:
        try:
            image_name = os.path.basename(path)
            crater_data = crater_info[crater_info['image_name'].str.lower() == image_name.lower()]
            if not crater_data.empty:
                num_craters = crater_data.iloc[0]['num_craters']
                label_path = os.path.join(LABELS_DIR, image_name.replace('.jpg', '.txt'))
                crater_details = []
                if os.path.exists(label_path):
                    with open(label_path, 'r') as label_file:
                        for line in label_file:
                            parts = line.strip().split()
                            crater_details.append({
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
                add_image_to_db(path, crater_details)
                images_added += 1
        except Exception as e:
            st.warning(f"Error adding image {path}: {e}")

    return images_added

# Preprocess text and find the best answer
def find_best_answer(question):
    # Fetch all text data
    c.execute("SELECT content FROM articles")
    articles = [row[0] for row in c.fetchall()]
    
    # Combine text data
    combined_text = articles
    
    # If no text available
    if not combined_text:
        return "No articles available for answering."
    
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(combined_text)
    
    # Vectorize the question
    question_vector = vectorizer.transform([question])
    
    # Compute cosine similarity
    similarities = cosine_similarity(question_vector, vectors)
    
    # Find the most similar document
    best_match_idx = np.argmax(similarities)
    best_match_text = combined_text[best_match_idx]
    
    # Perform question answering
    result = qa_pipeline(question=question, context=best_match_text)
    return result['answer']

# Find and display images based on a query
def find_image(query):
    num_craters = None
    words = re.findall(r'\b\d+\b', query)  # Extract numbers from the query
    if words:
        num_craters = int(words[0])
    
    if num_craters is not None:
        c.execute("SELECT id, file_path, image FROM images WHERE num_craters = ?", (num_craters,))
    else:
        c.execute("SELECT id, file_path, image FROM images")
    
    image_data = c.fetchall()

    if not image_data:
        return None
    
    images_found = {}
    for image_id, file_path, image_blob in image_data:
        try:
            image = Image.open(BytesIO(image_blob))
            c.execute("SELECT x_center, y_center, width, height FROM craters WHERE image_id = ?", (image_id,))
            crater_coords = c.fetchall()
            images_found[file_path] = (image, crater_coords)
        except Exception as e:
            st.warning(f"Error opening image {file_path}: {e}")

    return images_found

# Display image and crater coordinates
def display_images_and_craters(images_found):
    for file_path, (image, crater_coords) in images_found.items():
        st.image(image, caption=f"File Path: {file_path}")  
        if crater_coords:
            for coord in crater_coords:
                st.write(f"x_center: {coord[0]}, y_center: {coord[1]}, width: {coord[2]}, height: {coord[3]}")

# Add multiple articles from user input
def add_articles_from_input(article_urls):
    urls = [url.strip() for url in article_urls.split("\n") if url.strip()]
    for url in urls:
        fetch_article_text(url)

# Generate PDF report
def generate_pdf(question, answer):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Question and Answer Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Question: {question}", ln=True)
    pdf.cell(200, 10, txt=f"Answer: {answer}", ln=True)
    
    pdf_file_path = "report.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# Generate PDF from feedback
def generate_pdf_from_feedback(feedback):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Feedback Report", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=feedback)
    
    pdf_file_path = "feedback_report.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

# Streamlit UI
def main():
    st.title("Custom Dataset Query Tool")

    # Inject custom CSS
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
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ["Upload Local Image Dataset", "Add Articles", "Question Answering", "Query Images", "Feedback"])

    if options == "Upload Local Image Dataset":
        st.header("Upload Local Image Dataset")
        image_dir = st.text_input("Enter the directory path containing images:")

        if st.button("Process Images"):
            image_dir = image_dir.strip('"')  # Remove any extra quotes
            if os.path.isdir(image_dir):
                images_added = fetch_local_images_from_csv_and_labels(image_dir)
                st.success(f"Uploaded and processed {images_added} images")
            else:
                st.warning("The provided directory path is incorrect or does not exist.")
    
    elif options == "Add Articles":
        st.header("Add Articles")
        article_urls = st.text_area("Enter URLs of articles (one per line):")
        if st.button("Add Articles"):
            if article_urls:
                add_articles_from_input(article_urls)
                st.success("Articles have been added successfully.")
            else:
                st.warning("Please enter at least one article URL.")

    elif options == "Question Answering":
        st.header("Question Answering")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if question:
                answer = find_best_answer(question)
                st.write("Answer:", answer)
                
                # Generate and provide PDF download
                pdf_path = generate_pdf(question, answer)
                st.download_button("Download PDF Report", data=open(pdf_path, "rb").read(), file_name=pdf_path, mime="application/pdf")
            else:
                st.warning("Please enter a question.")

    elif options == "Query Images":
        st.header("Query Images")
        query = st.text_input("Enter your query (e.g., 'give me images with 5 craters'):")
        if st.button("Find Images"):
            if query:
                images_found = find_image(query)
                if images_found:
                    display_images_and_craters(images_found)
                else:
                    st.write("No images found matching the query.")
            else:
                st.warning("Please enter a query.")

    elif options == "Feedback":
        st.header("Feedback")
        feedback = st.text_area("Enter your feedback:")
        if st.button("Send Feedback"):
            if feedback:
                # Generate PDF report from feedback
                pdf_path = generate_pdf_from_feedback(feedback)
                
                # Provide download link for PDF
                st.download_button("Download Feedback PDF", data=open(pdf_path, "rb").read(), file_name=pdf_path, mime="application/pdf")
                
                # Encode feedback for mailto link
                encoded_feedback = urllib.parse.quote(feedback)
                mailto_link = f'mailto:chawlapc.619@gmail.com?subject=Feedback&body={encoded_feedback}'
                
                # Create and display the redirection link
                st.markdown(f'''
                <a href="{mailto_link}" target="_blank" id="email-link">Click here to send feedback via email</a>
                <script>
                    document.getElementById("email-link").click();
                </script>
                ''', unsafe_allow_html=True)
                
                st.success("Feedback PDF generated! You can download it and click the link to open your email client.")
            else:
                st.warning("Please enter your feedback.")

if __name__ == "__main__":
    main()
