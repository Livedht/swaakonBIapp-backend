import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
app.secret_key = os.environ.get('SECRET_KEY', 'your_fallback_secret_key')

nltk.download('punkt')
nltk.download('stopwords')

# Her kan det legges til flere stoppord. Har tatt ut liste over de 250 mest brukte ordene, og lagt til de jeg tenker ikke hører hjemme i analysen.
additional_stopwords = [
    'students', 'course', 'studentene', 'able', 'kunnskap', 'knowledge', 'understand', 'ulike', 'understanding', 'different', 'kurset', 'apply', 'skills', 'candidates', 'candidate', 'end', 'understands', 'forstår', 'able.', 'address',
' including', 'processes', 'kunne', 'practice', 'øve', 'ORG', 'EXC', 'ELE', 'BMP', 'GRA', 'BIK', 'BST', 'SMC', 'SLM', 'MAN', 'DRE', 'FORK', 'ENT', 'BØK', 'LUS', 'JUR', 'FAK', 'MET', 'EDI', 'EBA', 'FIN', 'KLS', 'STR', 'BTH', 'MRK', 'EMS', 'BIN', 'DIG', 'MAD', 'NSA', 'module', 'modul'  
]

EXCEL_FILE_PATH = os.path.join('data', 'Kombifil.xlsx')



 # This function cleans up the text by removing common but unimportant words (like "and", "the", etc.) in English and Norwegian.
def remove_stopwords(text, languages=['english', 'norwegian']):
    stop_words = set()
    for lang in languages:
        stop_words.update(set(stopwords.words(lang)))
        
    stop_words.update(additional_stopwords)

    word_tokens = word_tokenize(text)
    filtered_text = ' '.join(w for w in word_tokens if w.lower() not in stop_words)
    return filtered_text

 # Reads course data from an Excel file, organizing it so the application can understand and use it.
def load_courses_from_excel(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    courses = []
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    for row in sheet.iter_rows(min_row=2):
        course_data = {headers[i]: cell.value for i, cell in enumerate(row) if headers[i]}
        courses.append(course_data)
    return courses

# Cleans and prepares book titles or literature entries for comparison.
def normalize_literature_entry(entry):
    # Remove "Book: " prefix and any extra whitespace or quotes
    return set(entry.replace("Book: ", "").replace("'", "").replace('"', '').strip().lower().split('\n'))

 # Looks for any books or literature in the user's input that also appear in the course descriptions.
def find_literature_matches(user_input, existing_courses):
    user_titles = normalize_literature_entry(user_input)
    print(f"User titles after normalization: {user_titles}")

    matches = []
    for course in existing_courses:
        # Check if 'Pensum' key exists in the course dictionary
        if 'Pensum' in course and course['Pensum']:
            # Assuming 'Pensum' data is a string of titles separated by "|"
            course_titles = set(
                map(lambda x: x.strip().lower(), 
                    course['Pensum'].replace("Book: ", "").replace("'", "").split('|'))
            )
            print(f"Course: {course['Kursnavn']}, Normalized literature: {course_titles}")

            common_titles = user_titles.intersection(course_titles)
            print(f"Common titles found: {common_titles}")
            if common_titles:
                matches.append({
                    'Existing Course Code': course.get('Kurskode', 'N/A'),
                    'Existing Course Name': course.get('Kursnavn', 'Unknown'),
                    'Literature Matches': ' | '.join(f"Book: '{title}'" for title in common_titles)
                })
        else:
            # Handle the case where 'Pensum'(litteraturliste is not connected to the course) is not provided or is empty
            print(f"No 'Pensum' data for course: {course.get('Kursnavn', 'Unknown')}")
            
            matches.append({
                'Existing Course Code': course.get('Kurskode', 'N/A'),
                'Existing Course Name': course.get('Kursnavn', 'Unknown'),
                'Literature Matches': "No 'Pensum' data available"
            })

    print(f"Matches found: {matches}")
    return matches

 # Compares a new course description to existing ones to find similarities, helping to identify if the new course is too similar to existing ones.
def check_course_overlap(new_course_data, existing_courses, overlap_threshold=0.25):  # Overlap threshold can be adjusted. This is currently set to 25%. All results below 25 will not be included in the results
    for course in existing_courses:
        combined_info_fields = [
            str(course.get('Kurskode', '')), 
            str(course.get('Kursnavn', '')),
            str(course.get('Learning outcome - Knowledge', '')),
            str(course.get('Learning outcome - Skills', '')),
            str(course.get('Learning outcome - General Competence', '')),
            str(course.get('Course content', ''))
        ]
        combined_info = ' '.join(combined_info_fields).strip()
        course['combined_info'] = remove_stopwords(combined_info)

    new_course_combined_info_fields = [
        str(new_course_data.get('Kurskode', '')),
        str(new_course_data.get('Kursnavn', '')),
        str(new_course_data.get('Learning outcome - Knowledge', '')),
        str(new_course_data.get('Course content', ''))  # Combined input for learning outcomes and course content
    ]
    new_course_combined_info = ' '.join(new_course_combined_info_fields).strip()
    new_course_combined_info = remove_stopwords(new_course_combined_info)

    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    new_course_embedding = model.encode([new_course_combined_info], show_progress_bar=False)
    existing_courses_combined_info = [course['combined_info'] for course in existing_courses]
    existing_courses_embeddings = model.encode(existing_courses_combined_info, show_progress_bar=True)
    
    cosine_sim = cosine_similarity(new_course_embedding, existing_courses_embeddings)
    overlapping_courses = []

    for idx, sim_score in enumerate(cosine_sim[0]):
        if new_course_combined_info == existing_courses[idx]['combined_info']:
            overlapping_courses.append({
                'Existing Course Code': existing_courses[idx].get('Kurskode', 'N/A'),
                'Existing Course Name': existing_courses[idx]['Kursnavn'],
                'Overlap Score (%)': 100,
                'Keywords': 'Exact match'
            })
            continue

        if sim_score > overlap_threshold:
            sim_score_percentage = round(sim_score * 100, 2)
            # Extract keywords using RAKE
            r = Rake()
            r.extract_keywords_from_text(existing_courses[idx]['combined_info'])
            phrases_with_scores = r.get_ranked_phrases_with_scores()
            keywords = ', '.join(phrase for score, phrase in phrases_with_scores if score > 1)
            
            overlapping_courses.append({
                'Existing Course Code': existing_courses[idx].get('Kurskode', 'N/A'),
                'Existing Course Name': existing_courses[idx]['Kursnavn'],
                'Overlap Score (%)': sim_score_percentage,
                'Keywords': keywords  # Including keywords in the response
            })
    return overlapping_courses

# Load the existing courses from the Excel sheet when the application starts
existing_courses = load_courses_from_excel(EXCEL_FILE_PATH)

# Defines how the application should respond when someone visits the main page or sends information to it.
@app.route('/', methods=['GET', 'POST']) 
def home():
    if request.method == 'POST':
        new_course_name = request.form['new_course_name']
        learning_outcomes_and_content = request.form['learning_outcomes_and_content']
        literature = request.form.get('literature', '')  # Retrieve literature from form, if any
        
        print("Received literature data:", literature)

        new_course_details = {
            'Kurskode': 'N/A',  # Assuming Kurskode is not provided in the form
            'Kursnavn': new_course_name,
            'Learning outcome - Knowledge': learning_outcomes_and_content,  # Both learning outcomes and course content
            'Course content': learning_outcomes_and_content  # For consistency, assuming course content same as learning outcomes
        }
        overlapping_courses = check_course_overlap(new_course_details, existing_courses)

        literature_matches = []
        if literature:
            # Normalize the literature input from the form
            literature_input = "\n".join([line.strip() for line in literature.splitlines() if line.strip()])
            # Find matches based on the normalized literature input
            literature_matches = find_literature_matches(literature_input, existing_courses)

        # Extracting additional columns from existing courses
        additional_columns = [
            'Kurskode','Kurskode2','Academic Coordinator','Credits','Undv.språk',
            'Gj.føring','LINK EN','LINK NB','Level of study','Portfolio','Associate Dean',
            'Ansvarlig institutt','Ansvarlig område',
        ]
        additional_info = [{column: course.get(column, '') for column in additional_columns} for course in existing_courses]

        print("Literature Matches:", literature_matches)
        return jsonify({
            'overlapping_courses': overlapping_courses,
            'literature_matches': literature_matches,
            'additional_info': additional_info
        })

    else:
        return jsonify({'message': 'Ready for overlap checking'})

 # An additional feature that allows for the comparison of all courses to each other to find overlaps.
@app.route('/analyze_all_courses', methods=['GET'])
def analyze_all_courses():
    courses = existing_courses  # This assumes existing_courses already contains all necessary data

    # Make sure combined_info is computed and added here to avoid KeyError
    for course in courses:
        combined_info_fields = [
            str(course.get('Kurskode', '')), 
            str(course.get('Kursnavn', '')),
            str(course.get('Learning outcome - Knowledge', '')),
            str(course.get('Learning outcome - Skills', '')),
            str(course.get('Learning outcome - General Competence', '')),
            str(course.get('Course content', ''))
        ]
        combined_info = ' '.join(combined_info_fields).strip()
        course['combined_info'] = remove_stopwords(combined_info)

    # Proceed with encoding and the rest of the function...
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    all_embeddings = model.encode([course['combined_info'] for course in courses], show_progress_bar=True)

    results = []
    for i, course_a in enumerate(courses):
        for j, course_b in enumerate(courses):
            if i >= j:  # Avoid duplicates and self-comparison
                continue

            sim_score = cosine_similarity([all_embeddings[i]], [all_embeddings[j]])[0][0]
            sim_score_percentage = round(sim_score * 100, 2)
            
            if sim_score > 0.25:  # Assuming a threshold for a relevant comparison
                # Safely handle potentially None 'Pensum' values
                pensum_a = course_a.get('Pensum', '') or ''
                pensum_b = course_b.get('Pensum', '') or ''

                literature_a = set(pensum_a.split('|')) if pensum_a else set()
                literature_b = set(pensum_b.split('|')) if pensum_b else set()

                common_literature = literature_a.intersection(literature_b)
                common_literature_formatted = ' | '.join(common_literature)

                results.append({
                    'Course Code 1': course_a.get('Kurskode', 'N/A'),
                    'Course Name 1': course_a.get('Kursnavn', 'N/A'),
                    'Overlap Score (%)': sim_score_percentage,
                    'Course Code 2': course_b.get('Kurskode', 'N/A'),
                    'Course Name 2': course_b.get('Kursnavn', 'N/A'),
                    'Common Literature': common_literature_formatted
                })

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
