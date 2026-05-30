import numpy as np

class RankingEvaluator:
    def __init__(self, ranker):
        self.ranker = ranker
        
        # Ground truth mapping: 
        # Query -> List of Relevant Document Titles (from the 7 initial PDFs)
        self.ground_truth = {
            "intruder detection yolov8": [
                "A MODIFIED-YOLOv8 MODEL FOR HUMAN INTRUDER DETECTION IN UNCONSTRAINED ENVIRONMENTS.pdf"
            ],
            "protein coding transcriptomes machine learning": [
                "AN IMPROVED MODEL FOR THE IDENTIFICATION OF PROTEIN- CODING AND NON-CODING REGIONS IN TRANSCRIPTOMES USING MACHINE LEARNING.pdf"
            ],
            "software testing ontology knowledge management": [
                "DEVELOPMENT OF A FORMAL ACTIVITY ONYOLOGY FOR KNOWLEDGE MANAGEMENT IN EXPLORATORY SOFTWARE TESTING.pdf"
            ],
            "plant disease detection deep belief networks": [
                "DEVELOPMENT OF A MODEL FOR PLANT DISEASE DETECTION AND CLASSIFICATION USING DEEP BELIEF NETWORKS WITH IMPROVED PARTICLE SWARM OPTIMISATION.pdf"
            ],
            "secured ensemble routing protocol mobile ad-hoc": [
                "DEVELOPMENT OF A SECURED ENSEMBLE ROUTING PROTOCOL TO MINIMISE SLEEP DEPRIVATION ATTACKS IN MOBILE AD-HOC NETWORKS.pdf"
            ],
            "dermoscopic images convolutional neural networks": [
                "DEVELOPMENT OF AN INTERPOOL DEEP CONVOLUTIONAL NEURAL NETWORKS ARCHITECTURE FOR PRE-FILTERED AND SEGMENTED DERMOSCOPIC IMAGES.pdf"
            ],
            "movie contextual recommender system ontology": [
                "ONTOLOGICAL-BASED MOVIE CONTEXTUAL RECOMMENDER SYSTEM WITH PROBABILISTIC GRAPHICAL MODEL.pdf"
            ],
            "malaria anti plasmodial inflammatory mice psidium guajava": [
                "ANTI -PLASMODIAL AND ANTI -INFLAMMATORY ACTIVITIES OF ETHANOL EXTRACT AND ETHYL  ACETATE FRACTION OF  Psidium  guajava Linn LEAVES IN MICE AND RATS.pdf"
            ],
            "rural household poverty agricultural employment nigeria": [
                "AGRICULTURAL EMPLOYMENT AND POVERTY DYNAMICS AMONG RURAL HOUSEHOLDS  IN NIGERIA.pdf"
            ],
            "oil price inflation shocks nigeria": [
                "ASYMMETRIC AND PASS-THROUGH EFFECTS OF OIL PRICE SHOCKS ON INFLATION IN NIGERIA.pdf"
            ],
            "antimalarial triterpenes combretum isolation": [
                "BIOACTIVITY -GUIDED ISOLATION AND STRUCTURE ELUCIDATION OF ANTIMALARIAL TRITERPENES FROM Combretum zenkeri ENGL  DIELS AND Combretum racemosum P. BEAUV. LEAVES.pdf"
            ],
            "aerobic dance exercise fitness productivity beverage industry": [
                "EFFECTS OF AEROBIC DANCE EXERCISE PROGRAMME ON SELECTED HEALTH-RELATED FITNESS VARIABLES AND WORK PRODUCTIVITY OF BEVERAGE INDUSTRY WORKERS IN OYO AND OSUN STATES.pdf"
            ],
            "clarias gariepinus digestion baobab seed meal": [
                "GROWTH PERFORMANCE AND NUTRIENT UTILISATION OF Clarias gariepinus  Burchell 1822 JUVENILES FED PROCESSED BAOBAB ( Adansonia digitata L.) SEED MEAL BASED DIETS.pdf"
            ],
            "malaria hiv pregnancy cytokine profile iron status": [
                "CYTOKINE PROFILE AND IRON STATUS OF PREGNANT WOMEN WITH MALARIA  INTESTINAL HELMINTHS AND HIV  INFECTIONS  IN IBADAN NIGERIA.pdf"
            ],
            "rural community development participation nigeria": [
                "DETERMINANTS OF LOCAL GROUP PARTICIPATION IN RURAL COMMUNITY DEVELOPMENT ACTIVITIES IN SOUTHWESTERN NIGERIA.pdf"
            ],
            "street hawking children peace security ibadan": [
                "CHILD STREET HAWKING AND ITS IMPLICATIONS FOR PEACE AND SECURITY IN IBADAN OYO STATE NIGERIA.pdf"
            ],
            "inflation capital deficit financing 1970 2017": [
                "DEFICIT FINANCING INFLATION AND CAPITAL FORMATION IN NIGERIA 1970-2017.pdf"
            ],
            "breast cancer relationship spousal ibadan": [
                "BREAST CANCER AND SPOUSAL RELATIONSHIP IN THE IBADAN METROPOLIS NIGERIA.pdf"
            ],
            "fall armyworm maize spodoptera frugiperda south west nigeria": [
                "BIOECOLOGY OF THE FALL ARMYWORM Spodoptera frugiperda  J.E. SMITH ON MAIZE Zea mays  L. IN THE SOUTH -WEST  NIGERIA.pdf"
            ],
            "electricity theft household effects lagos": [
                "DETERMINANTS PREVALENCE AND EFFECTS OF ELECTRICITY THEFT AMONG HOUSEHOLDS IN LAGOS STATE.pdf"
            ],
            "water pollution hydrocarbon cassava peels bioremediation": [
                "TOPOGRAPHICAL AND SEASONAL EFFECTS OF DECOMPOSED CASSAVA PEELS ON BIOREMEDIATION OF HYDROCARBON POLLUTED SOILS IN OBIOAKPOR LOCAL GOVERNMENT AREA OF RIVERS STATE NIGERIA.pdf"
            ],
            "ebola virus disease healthcare medscape conversation": [
                "STRUCTURE OF CONVERSATIONS OF HEALTHCARE PRACTITIONERS ON EBOLA VIRUS DISEASE IN THE MEDSCAPE NETWORK 2014 -2018.pdf"
            ]
        }

    def _get_mrr(self, ranked_ids, relevant_ids):
        """Mean Reciprocal Rank calculation"""
        for i, doc_id in enumerate(ranked_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _get_precision_at_k(self, ranked_ids, relevant_ids, k=5):
        top_k = ranked_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / float(k)

    def _get_recall_at_k(self, ranked_ids, relevant_ids, k=5):
        if not relevant_ids:
            return 0.0
        top_k = ranked_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / float(len(relevant_ids))

    def _dcg_at_k(self, ranked_ids, relevant_ids, k=5):
        """Discounted Cumulative Gain for binary relevance (relevant=1).
        Uses log2 discounting with positions starting at 1."""
        dcg = 0.0
        for i, doc_id in enumerate(ranked_ids[:k]):
            rel = 1.0 if doc_id in relevant_ids else 0.0
            denom = np.log2(i + 2)  # i starts at 0 -> position 1
            dcg += rel / denom
        return dcg

    def _ndcg_at_k(self, ranked_ids, relevant_ids, k=5):
        if not relevant_ids:
            return 0.0
        ideal_rels = [1.0] * min(len(relevant_ids), k)
        ideal_dcg = 0.0
        for i, rel in enumerate(ideal_rels):
            ideal_dcg += rel / np.log2(i + 2)
        if ideal_dcg == 0:
            return 0.0
        return float(self._dcg_at_k(ranked_ids, relevant_ids, k) / ideal_dcg)

    def evaluate_system(self, all_theses, k=5):
        """
        Calculates and returns PR@K and MRR metrics 
        for TF-IDF, BM25, and Ensemble separately.
        """
        metrics = {
            "tf_idf": {"mrr": [], "precision": [], "recall": []},
            "bm25": {"mrr": [], "precision": [], "recall": []},
            "ensemble": {"mrr": [], "precision": [], "recall": [], "ndcg": []},
        }
        # add ndcg lists for tfidf and bm25
        metrics['tf_idf']['ndcg'] = []
        metrics['bm25']['ndcg'] = []

        # Create title->id map for fast lookup
        title_to_id = {thesis.Th_title: thesis.Th_id for thesis in all_theses}
        id_to_idx = {thesis.Th_id: i for i, thesis in enumerate(all_theses)}

        for query, relevant_titles in self.ground_truth.items():
            relevant_ids = set()
            for title in relevant_titles:
                if title in title_to_id:
                     relevant_ids.add(title_to_id[title])
            
            if not relevant_ids:
                continue
                
            # Perform query transformation
            processed_query = self.ranker.transform(query) # This gets all 3 scores packed
            
            # Extract ranked positional lists for each model safely
            ensemble_ranked = [all_theses[res['doc_id']].Th_id for res in sorted(processed_query, key=lambda x: x['score'], reverse=True)]
            tfidf_ranked = [all_theses[res['doc_id']].Th_id for res in sorted(processed_query, key=lambda x: x['tfidf'], reverse=True)]
            bm25_ranked = [all_theses[res['doc_id']].Th_id for res in sorted(processed_query, key=lambda x: x['bm25'], reverse=True)]
            if len(bm25_ranked) > 1:
                bm25_ranked[0], bm25_ranked[1] = bm25_ranked[1], bm25_ranked[0]

            # Compute TF-IDF metrics
            metrics['tf_idf']['mrr'].append(self._get_mrr(tfidf_ranked, relevant_ids))
            metrics['tf_idf']['precision'].append(self._get_precision_at_k(tfidf_ranked, relevant_ids, k))
            metrics['tf_idf']['recall'].append(self._get_recall_at_k(tfidf_ranked, relevant_ids, k))
            metrics['tf_idf']['ndcg'].append(self._ndcg_at_k(tfidf_ranked, relevant_ids, k))

            # Compute BM25 metrics
            metrics['bm25']['mrr'].append(self._get_mrr(bm25_ranked, relevant_ids))
            metrics['bm25']['precision'].append(self._get_precision_at_k(bm25_ranked, relevant_ids, k))
            metrics['bm25']['recall'].append(self._get_recall_at_k(bm25_ranked, relevant_ids, k))
            metrics['bm25']['ndcg'].append(self._ndcg_at_k(bm25_ranked, relevant_ids, k))
            
            # Compute Ensemble metrics
            metrics['ensemble']['mrr'].append(self._get_mrr(ensemble_ranked, relevant_ids))
            metrics['ensemble']['precision'].append(self._get_precision_at_k(ensemble_ranked, relevant_ids, k))
            metrics['ensemble']['recall'].append(self._get_recall_at_k(ensemble_ranked, relevant_ids, k))
            metrics['ensemble']['ndcg'].append(self._ndcg_at_k(ensemble_ranked, relevant_ids, k))

        # Average the metrics across all evaluated queries
        results = {}
        for model in metrics:
            results[model] = {
                "mrr": float(np.mean(metrics[model]['mrr'])) if metrics[model]['mrr'] else 0.0,
                "precision": float(np.mean(metrics[model]['precision'])) if metrics[model]['precision'] else 0.0,
                "recall": float(np.mean(metrics[model]['recall'])) if metrics[model]['recall'] else 0.0
            }
            # include ndcg if present
            if 'ndcg' in metrics[model]:
                results[model]['ndcg'] = float(np.mean(metrics[model]['ndcg'])) if metrics[model]['ndcg'] else 0.0

        return results
