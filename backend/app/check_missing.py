import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models.database import db, Theses

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/theses_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    theses = Theses.query.filter(Theses.Th_year.is_(None)).all()
    print(f"Total without a year: {len(theses)}")
    for i, t in enumerate(theses, 1):
        print(f"{i}. {t.Th_title}")
