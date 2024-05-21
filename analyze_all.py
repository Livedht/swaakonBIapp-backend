@app.route('/analyze_all_courses', methods=['GET'])
def analyze_all_courses():
    courses = existing_courses

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