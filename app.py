import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sqlite3
import re
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
app.secret_key = os.environ.get('SECRET_KEY', 'your_fallback_secret_key')

nltk.download('punkt')
nltk.download('stopwords')

# Her kan det legges til flere stoppord. Har tatt ut liste over de 250 mest brukte ordene, og lagt til de jeg tenker ikke hører hjemme i analysen.
additional_stopwords = [
    'students', 'student', 'course', 'studentene', 'able', 'kunnskap', 'knowledge', 'understand', 'ulike',
    'understanding', 'different', 'kurset', 'apply', 'skills', 'candidates', 'candidate', 'end', 'understands',
    'forstår', 'able.', 'address', ' including', 'processes', 'kunne', 'practice', 'øve', 'org', 'exc', 'ele',
    'bmp', 'gra', 'bik', 'bst', 'smc', 'slm', 'man', 'dre', 'fork', 'ent', 'bøk', 'lus', 'jur', 'fak', 'met',
    'edi', 'eba', 'fin', 'kls', 'str', 'bth', 'mrk', 'ems', 'bin', 'dig', 'mad', 'nsa', 'module', 'modul'
]

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# Global variable for caching
cache = {}

def compute_embeddings(text):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    embeddings = model.encode([text], show_progress_bar=False)
    return embeddings

def remove_stopwords(text, languages=['english', 'norwegian']):
    stop_words = set()
    for lang in languages:
        stop_words.update(set(stopwords.words(lang)))

    stop_words.update(additional_stopwords)

    word_tokens = word_tokenize(text)
    filtered_text = ' '.join(w for w in word_tokens if w.lower() not in stop_words and not re.search(r'\b\d+\b', w))
    return filtered_text

def load_courses_from_database():
    conn = sqlite3.connect('data/course_db.sqlite')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Kombifildb")
    courses = []
    for row in cursor.fetchall():
        embeddings_hex = row['Embeddings']
        if embeddings_hex:
            try:
                # Attempt to decode bytes using UTF-8
                embeddings_hex_str = embeddings_hex.decode('utf-8')
            except UnicodeDecodeError:
                # If decoding using UTF-8 fails, try another encoding or ignore errors
                embeddings_hex_str = embeddings_hex.decode('latin-1', errors='ignore')

            try:
                # Convert hexadecimal string to bytes
                embeddings_bytes = bytes.fromhex(embeddings_hex_str)
                # Assuming the embedding size, e.g., 512 dimensions of float32
                # Adjust the size according to your actual embedding dimensions
                embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32).reshape(-1, 512)
            except ValueError:
                # If non-hexadecimal characters are present, set embeddings to None
                embeddings = None
        else:
            embeddings = None

        course_data = {
            'Kurskode': row['Kurskode'],
            'Kursnavn': row['Kursnavn'],
            'Learning outcome - Knowledge': row['Learningoutcome-Knowledge'],
            'Learning outcome - Skills': row['Learningoutcome-Skills'],
            'Learning outcome - General Competence': row['Learningoutcome-GeneralCompetence'],
            'Course content': row['Coursecontent'],
            'Pensum': row['Pensum'],
            'Keywords': row['Keywords'],
            'Embeddings': embeddings
        }
        courses.append(course_data)
    conn.close()
    return courses

# Load the existing courses from the SQLite database when the application starts
existing_courses = load_courses_from_database()

def normalize_literature_entry(entry):
    return set(entry.replace("Book: ", "").replace("'", "").replace('"', '').strip().lower().split('\n'))

def find_literature_matches(user_input, existing_courses):
    user_titles = normalize_literature_entry(user_input)

    matches = []
    for course in existing_courses:
        if 'Pensum' in course and course['Pensum']:
            course_titles = set(
                map(lambda x: x.strip().lower(),
                    course['Pensum'].replace("Book: ", "").replace("'", "").split('|'))
            )

            common_titles = user_titles.intersection(course_titles)
            if common_titles:
                matches.append({
                    'Existing Course Code': course.get('Kurskode', 'N/A'),
                    'Existing Course Name': course.get('Kursnavn', 'Unknown'),
                    'Literature Matches': ' | '.join(f"Book: '{title}'" for title in common_titles)
                })
        else:
            matches.append({
                'Existing Course Code': course.get('Kurskode', 'N/A'),
                'Existing Course Name': course.get('Kursnavn', 'Unknown'),
                'Literature Matches': "No 'Pensum' data available"
            })

    return matches

def check_course_overlap(new_course_data, existing_courses, overlap_threshold=0.25):
    new_course_combined_info = ' '.join([
        str(new_course_data.get('Kurskode', '')),
        str(new_course_data.get('Kursnavn', '')),
        str(new_course_data.get('Learning outcome - Knowledge', '')),
        str(new_course_data.get('Course content', ''))
    ]).strip()
    new_course_combined_info = remove_stopwords(new_course_combined_info)

    new_course_embedding = model.encode([new_course_combined_info], show_progress_bar=False)

    overlapping_courses = []

    for existing_course in existing_courses:
        existing_course_embedding = existing_course['Embeddings']
        sim_score = cosine_similarity(new_course_embedding, existing_course_embedding)[0][0]

        if sim_score > overlap_threshold:
            sim_score_percentage = round(sim_score * 100, 2)

            overlapping_courses.append({
                'Existing Course Code': existing_course.get('Kurskode', 'N/A'),
                'Existing Course Name': existing_course['Kursnavn'],
                'Overlap Score (%)': sim_score_percentage,
            })

    return overlapping_courses

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        new_course_name = request.form['new_course_name']
        learning_outcomes_and_content = request.form['learning_outcomes_and_content']
        literature = request.form.get('literature', '')  # Retrieve literature from form, if any

        # Compute keywords and embeddings for the new course
        filtered_text = remove_stopwords(learning_outcomes_and_content)
        r = Rake()
        r.extract_keywords_from_text(filtered_text)
        keywords = ', '.join(r.get_ranked_phrases())  # Assuming get_ranked_phrases returns an iterable

        new_course_embeddings = compute_embeddings(filtered_text)

        new_course_details = {
            'Kurskode': 'N/A',  # Assuming Kurskode is not provided in the form
            'Kursnavn': new_course_name,
            'Learning outcome - Knowledge': learning_outcomes_and_content,  # Both learning outcomes and course content
            'Course content': learning_outcomes_and_content,  # For consistency, assuming course content same as learning outcomes
            'Keywords': keywords,
            'Embeddings': new_course_embeddings
        }

        overlapping_courses = check_course_overlap(new_course_details, existing_courses)

        literature_matches = []
        if literature:
            literature_input = "\n".join([line.strip() for line in literature.splitlines() if line.strip()])
            literature_matches = find_literature_matches(literature_input, existing_courses)

        return jsonify({
            'overlapping_courses': overlapping_courses,
            'literature_matches': literature_matches,
        })

    else:
        return jsonify({'message': 'Ready for overlap checking'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

