
## Development of a context-aware NLP system that can understand and retrieve geospatial data based on the context provided in user queries. The system is able to infer implicit information and provide relevant geospatial data accordingly, along with a specific use case of Mars/Moon Crater Detection.


## OUTPUT
![image](https://github.com/user-attachments/assets/59b7802e-1311-41a8-825a-4ceb87fafe24)
![image](https://github.com/user-attachments/assets/e6fb8b7d-5ff9-4cf2-8c40-5638cbd4c894)

![image](https://github.com/user-attachments/assets/547d2f41-bbf4-4b1d-9980-e48b6bfd4209)

![image](https://github.com/user-attachments/assets/20fb5056-b1a4-4977-abed-b344d87b795b)


## Explaination of the Repositories
Images directory - All the training images
Labels - Containing text files for all the coordinates of the craters
Output - Directory containing the images, having detected craters and visualized them.
test - All the testing images along with Labels
valid -  validation dataset
Marc crater info.csv - CSV file containing the mars crater info and names
detection.ipynb - Model training for the complete  crater detection process
final_app.py - The whole exact LLM app structure.
labels.csv - csv containing all the data, like number of craters and independent id of all the images
mars.ipynb - The file having the complete modelling of mars craters, its detection and its naming.
requirements.txt - Requirements of the project.


## Work plan
Phase 1: Initial Setup and Planning
Requirements Gathering
Identify all the functional and non-functional requirements.
Define the scope and objectives of the project.
Gather all necessary datasets and resources (images, articles, metadata).
Project Setup

Set up version control with a repository (e.g., GitHub).
Create a virtual environment and initialize the project.
Create the initial structure for the project.
Phase 2: Development
Database Setup

Design and set up the SQLite database schema for storing images and articles.
Implement database connection and setup functions.
Image Management

Develop functions to upload, process, and store local images and metadata.
Implement functionality to fetch and process images from the specified directories.
Article Management

Implement functions to fetch and parse article content from URLs.
Store fetched article content in the database.
Question Answering System

Integrate the NLP model for question answering.
Develop functionality to process user questions and fetch relevant answers from the stored articles.
Image Querying

Implement search functionality to find and display images based on metadata.
Develop functions to display images and associated crater coordinates.
Feedback Collection

Create a feedback form to collect user input.
Implement PDF generation for feedback reports.
Develop functionality to redirect users to their email client with pre-filled feedback content.

## TF IDF Vector

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. It is widely used in text mining and information retrieval. TF-IDF is composed of two main components: Term Frequency (TF) and Inverse Document Frequency (IDF). TF measures how frequently a term appears in a specific document, emphasizing terms that are more common within that document. IDF, on the other hand, assesses the importance of a term across the entire document corpus by reducing the weight of terms that appear in many documents and increasing the weight of terms that are rare. By combining these two metrics, TF-IDF helps to identify terms that are both significant in individual documents and unique across the corpus, making it an effective tool for various natural language processing tasks, such as keyword extraction, document similarity, and information retrieval.




## Cosine similarity


Cosine similarity is a metric used to measure the similarity between two vectors in a multi-dimensional space. It is widely used in text analysis and information retrieval to determine how similar two documents are, regardless of their size. Cosine similarity is calculated by taking the dot product of the vectors and dividing it by the product of their magnitudes. This results in a value between -1 and 1, where 1 indicates that the vectors are identical, 0 indicates no similarity, and -1 indicates that they are diametrically opposed. In the context of text analysis, cosine similarity can be used to compare documents represented as TF-IDF vectors, allowing for the identification of documents with similar content based on the angle between their vector representations, rather than their absolute differences in word counts. This makes it particularly useful for tasks such as document clustering, recommendation systems, and semantic search.







## Libraries and Usage

```
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






## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is used a lot of Research based companies, organizations and research centres
## Appendix

A very crucial project in the realm of LLM,data science and new age predictions domain using visualization techniques as well as machine learning modelling.

## Tech Stack

**Client:** Python
Streamlit: For building the interactive web interface.
Requests: For fetching article content from URLs.
BeautifulSoup: For parsing HTML content of articles.
SQLite: For managing and querying the database of images and articles.
Pandas: For handling CSV files and data processing.
Transformers (Hugging Face): For question-answering functionality.
FPDF: For generating PDF reports.
PIL (Pillow): For handling and displaying images.
NumPy: For numerical operations and similarity calculations.
Scikit-learn: For vectorizing text and calculating cosine similarity.
HTML, CSS 




## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

