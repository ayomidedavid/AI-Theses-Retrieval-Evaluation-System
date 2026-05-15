import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models.database import db, Theses
from utils.preprocessing import TextPreprocessor

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/theses_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

preprocessor = TextPreprocessor()

with app.app_context():
    theses = Theses.query.filter(Theses.Th_year.is_(None)).all()
    count = 0
    total = len(theses)
    
    print(f"Found {total} records needing year extraction...")
    
    for t in theses:
        file_path = t.Th_file_path
        if os.path.exists(file_path):
            raw_text = preprocessor.extract_text_from_pdf(file_path)
            if raw_text:
                year = preprocessor.extract_year(raw_text)
                if year:
                    t.Th_year = year
                    count += 1
            
    db.session.commit()
    print(f"Successfully extracted and updated {count} out of {total} theses.")
