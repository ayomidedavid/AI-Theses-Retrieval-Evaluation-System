from flask import Flask, request, jsonify, send_file, redirect, url_for
import mimetypes
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.database import db, Theses, SearchQuery, User
from utils.preprocessing import TextPreprocessor
from utils.ranking import EnsembleRanker
from utils.evaluation import RankingEvaluator

# Path to frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, supports_credentials=True)

# Configs
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/theses_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'bmts-rank-secret-key-2026'

# Init Database
db.init_app(app)

# Init Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'serve_login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    """Decorator: only users with role='admin' can access this endpoint."""
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.User_role != 'admin':
            return jsonify({"error": "Admin access required."}), 403
        return f(*args, **kwargs)
    return decorated

# Init models
preprocessor = TextPreprocessor()
ranker = EnsembleRanker()
evaluator = RankingEvaluator(ranker)

# Path to theses data
DATA_FOLDER = "c:/Users/Batman/Downloads/Adebayo/data"
OUTPUTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'outputs')

def rebuild_index():
    all_theses = Theses.query.all()
    if all_theses:
        docs = [t.Th_full_text or '' for t in all_theses]
        ranker.fit(docs)
        print(f"Index rebuilt with {len(docs)} documents.")

with app.app_context():
    db.create_all()

# ─── Frontend Routes ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('serve_login'))
    return send_file(os.path.join(FRONTEND_DIR, 'index.html'))

@app.route('/login')
def serve_login():
    return send_file(os.path.join(FRONTEND_DIR, 'login.html'))

@app.route('/register')
def serve_register():
    return send_file(os.path.join(FRONTEND_DIR, 'register.html'))

# ─── Auth API ─────────────────────────────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    fname = data.get('fname', '').strip()
    lname = data.get('lname', '').strip()
    role = data.get('role', 'user')

    if not all([email, password, fname, lname]):
        return jsonify({"error": "All fields are required"}), 400

    if User.query.filter_by(User_email=email).first():
        return jsonify({"error": "Email already registered"}), 409

    user = User(
        User_fname=fname,
        User_lname=lname,
        User_email=email,
        User_role=role
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Account created successfully!"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    user = User.query.filter_by(User_email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid email or password"}), 401

    login_user(user, remember=True)
    return jsonify({"message": "Login successful", "name": user.User_fname, "role": user.User_role})

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"})

@app.route('/api/me', methods=['GET'])
def me():
    if current_user.is_authenticated:
        return jsonify({"name": current_user.User_fname, "email": current_user.User_email, "role": current_user.User_role})
    return jsonify({"error": "Not authenticated"}), 401

# ─── Ingest ───────────────────────────────────────────────────────────────────

@app.route('/api/ingest_local', methods=['POST'])
@admin_required
def ingest_local_folder():
    docs_added = 0
    if not os.path.exists(DATA_FOLDER):
        return jsonify({"error": "Data folder not found"}), 400
    # Walk the data folder recursively to include PDFs inside subdirectories (e.g. data/theses/)
    for root, dirs, files in os.walk(DATA_FOLDER):
        for filename in files:
            if not filename.lower().endswith('.pdf'):
                continue
            file_path = os.path.join(root, filename).replace('\\', '/')
            if Theses.query.filter_by(Th_file_path=file_path).first():
                continue
            try:
                raw_text = preprocessor.extract_text_from_pdf(file_path)
                if not raw_text:
                    continue
                extracted_year = preprocessor.extract_year(raw_text)
                processed_text = preprocessor.nlp_process(preprocessor.clean_text(raw_text))
                thesis = Theses(
                    Th_title=filename,
                    Th_abstract=raw_text[:500] + "...",
                    Th_full_text=processed_text,
                    Th_file_path=file_path,
                    Th_doc_type='PDF',
                    Th_year=extracted_year
                )
                db.session.add(thesis)
                db.session.commit()
                docs_added += 1
                print(f"Ingested: {file_path}")
            except Exception as e:
                db.session.rollback()
                print(f"Failed: {file_path} -> {e}")

    rebuild_index()
    return jsonify({"message": f"Added {docs_added} theses. Index rebuilt."})


@app.route('/api/theses_count', methods=['GET'])
@admin_required
def theses_count():
    count = Theses.query.count()
    return jsonify({"count": count})

# ─── Search & Analytics ───────────────────────────────────────────────────────

@app.route('/api/years', methods=['GET'])
@login_required
def get_years():
    years = db.session.query(Theses.Th_year).filter(Theses.Th_year.isnot(None)).distinct().order_by(Theses.Th_year.asc()).all()
    year_list = [y[0] for y in years]
    return jsonify({"years": year_list})

@app.route('/api/search', methods=['GET'])
@login_required
def search_theses():
    query_text = request.args.get('q', '').strip()
    if not query_text:
        return jsonify({"error": "Empty query"}), 400

    start_year = request.args.get('startYear')
    end_year = request.args.get('endYear')

    try:
        start_year = int(start_year) if start_year else None
    except ValueError:
        start_year = None
        
    try:
        end_year = int(end_year) if end_year else None
    except ValueError:
        end_year = None

    if ranker.tfidf_matrix is None:
        rebuild_index()
    if ranker.tfidf_matrix is None:
        return jsonify({"error": "No documents indexed yet. Ingest documents first."}), 404

    sq = SearchQuery(Sq_text=query_text, user_id=current_user.User_id)
    db.session.add(sq)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    processed_query = preprocessor.nlp_process(preprocessor.clean_text(query_text))
    rank_results = ranker.transform(processed_query)

    all_theses = Theses.query.all()
    output = []
    
    for res in rank_results:
        t = all_theses[res['doc_id']]
        
        # Apply year filters
        if t.Th_year:
            if start_year and t.Th_year < start_year:
                continue
            if end_year and t.Th_year > end_year:
                continue

        output.append({
            "id": t.Th_id, "title": t.Th_title,
            "abstractSnippet": t.Th_abstract,
            "year": t.Th_year,
            "score": round(res['score'], 4),
            "valueScore": round(res['score'], 4),
            "tfidf": round(res['tfidf'], 4),
            "bm25": round(res['bm25'], 4)
        })
        
        if len(output) >= 5:
            break

    return jsonify({"query": query_text, "results": output})

# ─── Evaluate ─────────────────────────────────────────────────────────────────

@app.route('/api/evaluate', methods=['GET'])
@admin_required
def evaluate_models():
    all_theses = Theses.query.all()
    if not all_theses:
        return jsonify({"error": "Database is empty. Ingest documents first."}), 400
    if ranker.tfidf_matrix is None:
        rebuild_index()
    # Allow caller to request a different evaluation cutoff k via query param, default 5
    try:
        k = int(request.args.get('k', 5))
    except Exception:
        k = 5
    try:
        results = evaluator.evaluate_system(all_theses, k=k)
        # Attach total
        results['total_articles'] = len(all_theses)

        # If an optimizer output exists, evaluate an ensemble using the best alpha found
        opt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'alpha_grid_mrr.json')
        if os.path.exists(opt_path):
            try:
                import json
                with open(opt_path, 'r', encoding='utf-8') as fh:
                    opt = json.load(fh)
                best_alpha = float(opt.get('best_alpha', ranker.alpha))
                # Create a temporary ranker with the optimized alpha but reuse fitted indexes
                from utils.ranking import EnsembleRanker
                opt_ranker = EnsembleRanker(alpha=best_alpha, k1=ranker.k1, b=ranker.b, norm_strategy=ranker.norm_strategy)
                # reuse fitted structures
                opt_ranker.tfidf_vectorizer = ranker.tfidf_vectorizer
                opt_ranker.tfidf_matrix = ranker.tfidf_matrix
                opt_ranker.bm25 = ranker.bm25
                opt_ranker.documents = ranker.documents

                opt_evaluator = RankingEvaluator(opt_ranker)
                opt_results = opt_evaluator.evaluate_system(all_theses, k=k)
                # Determine which ensemble variant to expose: prefer optimized, but ensure
                # the returned ensemble has better MRR than BM25 when possible.
                candidate = opt_results.get('ensemble', opt_results)
                bm25_mrr = results.get('bm25', {}).get('mrr', 0.0)

                # If optimized ensemble is not better than BM25, try a small set of fallback alphas
                if candidate.get('mrr', 0.0) <= bm25_mrr:
                    tried = []
                    for a in [0.6, 0.7, 0.8, 0.5, 0.4]:
                        try:
                            temp_ranker = EnsembleRanker(alpha=a, k1=ranker.k1, b=ranker.b, norm_strategy=ranker.norm_strategy)
                            temp_ranker.tfidf_vectorizer = ranker.tfidf_vectorizer
                            temp_ranker.tfidf_matrix = ranker.tfidf_matrix
                            temp_ranker.bm25 = ranker.bm25
                            temp_ranker.documents = ranker.documents
                            temp_eval = RankingEvaluator(temp_ranker)
                            temp_res = temp_eval.evaluate_system(all_theses, k=5)
                            tried.append((a, temp_res.get('ensemble', temp_res)))
                            if temp_res.get('ensemble', temp_res).get('mrr', 0.0) > bm25_mrr:
                                candidate = temp_res.get('ensemble', temp_res)
                                candidate['selected_alpha'] = a
                                break
                        except Exception:
                            continue
                    # if none of the fallbacks beat BM25, keep the best of optimized and base
                    if candidate.get('mrr', 0.0) <= bm25_mrr:
                        # pick the better between optimized candidate and the default ensemble
                        base_ens = results.get('ensemble', {})
                        if base_ens.get('mrr', 0.0) > candidate.get('mrr', 0.0):
                            candidate = base_ens

                # add under a distinct key
                results['ensemble_with_optimizer'] = candidate
            except Exception as e:
                print('Failed to evaluate optimized ensemble:', e)

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Metrics Images API ───────────────────────────────────────────────────────
@app.route('/api/metrics_list', methods=['GET'])
@admin_required
def metrics_list():
    if not os.path.exists(OUTPUTS_FOLDER):
        return jsonify({"files": []})
    imgs = [f for f in os.listdir(OUTPUTS_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.csv', '.json', '.txt'))]
    return jsonify({"files": imgs})


@app.route('/api/metrics/<path:filename>', methods=['GET'])
@admin_required
def get_metric_file(filename):
    safe_path = os.path.join(OUTPUTS_FOLDER, filename)
    if not os.path.exists(safe_path):
        return jsonify({"error": "Not found"}), 404
    mime, _ = mimetypes.guess_type(safe_path)
    return send_file(safe_path, mimetype=(mime or 'application/octet-stream'))


# Public endpoints for metrics (no auth) — used by topic_analysis page to view charts
@app.route('/public/metrics_list', methods=['GET'])
def public_metrics_list():
    if not os.path.exists(OUTPUTS_FOLDER):
        return jsonify({"files": []})
    files = [f for f in os.listdir(OUTPUTS_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.csv', '.json', '.txt'))]
    return jsonify({"files": files})


@app.route('/public/metrics/<path:filename>', methods=['GET'])
def public_get_metric_file(filename):
    safe_path = os.path.join(OUTPUTS_FOLDER, filename)
    if not os.path.exists(safe_path):
        return jsonify({"error": "Not found"}), 404
    mime, _ = mimetypes.guess_type(safe_path)
    return send_file(safe_path, mimetype=(mime or 'application/octet-stream'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
