Course Analysis Tool
The Course Analysis Tool is a Python application designed to analyze course descriptions and literature lists to identify overlaps and similarities between courses. It utilizes natural language processing techniques and machine learning models to provide insights into course content and literature requirements.

Features
- Overlap Analysis: Compare new course descriptions with existing ones to identify potential overlaps in content.
- Literature Matching: Find matches between literature lists of different courses to identify common resources.
- Data Visualization: Visualize overlap scores and literature matches for easy interpretation.

Installation
1. Clone the Repository: Clone the repository to your local machine using the following command:
bash

git clone <repository_url>

2. Navigate to the Project Directory: Move into the project directory:


cd course-analysis-tool

3. Install Dependencies: Ensure you have Python installed. Then, install the required dependencies listed in the requirements.txt file using pip:

pip install -r requirements.txt

4. Set Up Environment Variables (if necessary): If your application requires any environment variables to run, ensure they are set up accordingly. You may need to create a .env file in the project directory to store sensitive information.

Usage
Run the Application: Start the application by executing the main script:

python script.py

Access the Application: Once the application is running, you can access it via a web browser at http://localhost:5000.
Input Course Details: Enter the relevant course details and literature lists into the provided fields for analysis.


Dependencies:
The project relies on the following Python packages, which are listed in the requirements.txt file:

Flask
Flask-CORS
NumPy
Sentence Transformers
scikit-learn
RAKE-NLTK
NLTK
openpyxl


Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.