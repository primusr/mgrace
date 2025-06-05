from flask import Flask, render_template, request
import os
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    feedback_summary = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process CSV
            df = pd.read_csv(filepath)
            
            # Example processing: count responses per column
            feedback_summary = df.describe(include='all').to_html(classes='table table-striped')

    return render_template('index.html', feedback_summary=feedback_summary)

if __name__ == '__main__':
    app.run(debug=True)
