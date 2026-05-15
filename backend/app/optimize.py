import optuna
import os
import sys

# Ensure backend package can be accessed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models.database import db, Theses
from utils.ranking import EnsembleRanker
from utils.evaluation import RankingEvaluator

# Minimal Flask app to use SQLAlchemy Context
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/theses_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def objective(trial):
    # Retrieve all theses within context
    with app.app_context():
        all_theses = Theses.query.all()
        if not all_theses:
            raise ValueError("No data in database to run optimization. Please ingest data first.")
        
        # We need to extract the processed text we run ranking on.
        # Ensure we avoid None texts
        docs = [t.Th_full_text or '' for t in all_theses]

        # 1. Define hyperparameters to search
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        k1 = trial.suggest_float("k1", 1.0, 3.0)
        b = trial.suggest_float("b", 0.0, 1.0)
        norm_strategy = trial.suggest_categorical("norm_strategy", ["minmax", "max", "l1", "l2", "zscore"])

        # 2. Instantiate ranker and rank evaluator
        ranker = EnsembleRanker(alpha=alpha, k1=k1, b=b, norm_strategy=norm_strategy)
        ranker.fit(docs)
        
        evaluator = RankingEvaluator(ranker)

        # 3. Evaluate the metrics specifically for ensemble search
        try:
            results = evaluator.evaluate_system(all_theses, k=5)
            # Maximize the Mean Reciprocal Rank (MRR) of the ensemble model
            ensemble_mrr = results['ensemble'].get('mrr', 0.0)
            return ensemble_mrr
        except Exception as e:
            # If a strategy fails (e.g., division by zero that we didn't catch), treat it as bad trial
            return 0.0

if __name__ == "__main__":
    print("Starting Optuna TPE Optimization...")
    
    # Optuna defaults to TPE (Tree-structured Parzen Estimator)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="BMTS-Rank-Optimization")
    
    # Run 50 trials
    study.optimize(objective, n_trials=50)

    print("\noptimization finished!")
    print("Best Trial Info:")
    trial = study.best_trial

    print(f"  Value (MRR): {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
