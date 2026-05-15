import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models.database import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/theses_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

with app.app_context():
    try:
        # Check if column exists, if not create it
        with db.engine.connect() as conn:
            conn.execute(db.text("ALTER TABLE theses ADD COLUMN Th_year INT NULL;"))
            print("Successfully added Th_year column to theses table.")
    except Exception as e:
        print("Column might already exist or another error occurred:", str(e))
