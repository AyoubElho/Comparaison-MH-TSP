"""
experiment.py
Exécute toutes les métaheuristiques sur toutes les instances du TSP et produit :
  - Un tableau récapitulatif des résultats (CSV + affiché)
  - Des graphiques des courbes de convergence (PNG)
  - Des données brutes par exécution (JSON)

Utilisation :
    python experiment.py
"""

import os
import sys
import json
import time
import random
import statistics
import csv

# Permet l'importation depuis la racine du projet
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_instances import (
    generate_random_instance,
    build_distance_matrix,
    save_instance,
    load_instance,
)
from algorithms import (
    random_tour,
    tour_length,
    hill_climbing_best,
    hill_climbing_first,
    multi_start_hill_climbing,
    simulated_annealing,
    tabu_search,
    grasp,
)

# ─── Tente d'importer matplotlib (optionnel pour les graphiques) ───
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[ATTENTION] matplotlib introuvable – les graphiques seront ignorés.")

# ══════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════

INSTANCES = [
    {"name": "Instance_A_20", "n": 20, "seed": 42},
    {"name": "Instance_B_50", "n": 50, "seed": 123},
    {"name": "Instance_C_80", "n": 80, "seed": 999},
]

N_RUNS = 30           # exécutions indépendantes par algorithme et par instance
MAX_EVALS = 200_000   # budget d'évaluations partagé entre les algorithmes

# Paramètres du Recuit Simulé (SA)
SA_T0    = 1000.0
SA_ALPHA = 0.995
SA_TMIN  = 1e-3

# Recherche locale à départs multiples (MS-HC)
MS_STARTS = 30

# Recherche Tabou
TABU_TENURE = 10

# GRASP
GRASP_ITERATIONS = 30
GRASP_ALPHA      = 0.3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
#  FONCTIONS AUXILIAIRES
# ══════════════════════════════════════════════════════

def ensure_instance(inst_cfg):
    """Charge ou génère+sauvegarde une instance du TSP, retourne (villes, matrice_distances)."""
    filepath = os.path.join(DATA_DIR, inst_cfg["name"] + ".json")
    if not os.path.exists(filepath):
        cities = generate_random_instance(inst_cfg["n"], seed=inst_cfg["seed"])
        save_instance(cities, filepath)
    else:
        cities = load_instance(filepath)
    dist = build_distance_matrix(cities)
    return cities, dist


def run_once(algorithm_name, n, dist, seed):
    """Exécute une instance de l'algorithme avec une graine aléatoire donnée. Retourne (coût, temps_s, convergence)."""
    random.seed(seed)
    start_t = time.perf_counter()
    convergence = None

    if algorithm_name == "HC_Best":
        init = random_tour(n)
        tour, cost, _ = hill_climbing_best(init, dist, max_evals=MAX_EVALS)

    elif algorithm_name == "HC_First":
        init = random_tour(n)
        tour, cost, _ = hill_climbing_first(init, dist, max_evals=MAX_EVALS)

    elif algorithm_name == "HC_Best_2opt":
        init = random_tour(n)
        tour, cost, _ = hill_climbing_best(init, dist, max_evals=MAX_EVALS, use_2opt=True)

    elif algorithm_name == "HC_First_2opt":
        init = random_tour(n)
        tour, cost, _ = hill_climbing_first(init, dist, max_evals=MAX_EVALS, use_2opt=True)

    elif algorithm_name == "MS_HC":
        tour, cost, _, convergence = multi_start_hill_climbing(
            n, dist, n_starts=MS_STARTS, hc_variant='best',
            max_evals=MAX_EVALS)

    elif algorithm_name == "SA":
        init = random_tour(n)
        tour, cost, _, convergence = simulated_annealing(
            init, dist, T0=SA_T0, alpha=SA_ALPHA, T_min=SA_TMIN,
            max_evals=MAX_EVALS)

    elif algorithm_name == "SA_2opt":
        init = random_tour(n)
        tour, cost, _, convergence = simulated_annealing(
            init, dist, T0=SA_T0, alpha=SA_ALPHA, T_min=SA_TMIN,
            max_evals=MAX_EVALS, use_2opt=True)

    elif algorithm_name == "Tabu":
        init = random_tour(n)
        tour, cost, _, convergence = tabu_search(
            init, dist, tabu_tenure=TABU_TENURE, max_evals=MAX_EVALS)

    elif algorithm_name == "GRASP":
        tour, cost, _, convergence = grasp(
            n, dist, n_iterations=GRASP_ITERATIONS, alpha_rcl=GRASP_ALPHA,
            hc_variant='best', max_evals=MAX_EVALS)

    else:
        raise ValueError(f"Algorithme inconnu : {algorithm_name}")

    elapsed = time.perf_counter() - start_t
    return cost, elapsed, convergence


# ══════════════════════════════════════════════════════
#  BOUCLE D'EXPÉRIMENTATION PRINCIPALE
# ══════════════════════════════════════════════════════

ALGORITHMS = [
    "HC_Best",
    "HC_First",
    "HC_Best_2opt",
    "HC_First_2opt",
    "MS_HC",
    "SA",
    "SA_2opt",
    "Tabu",
    "GRASP",
]

all_results = []   # liste de dictionnaires pour le CSV
raw_data    = {}   # données brutes par exécution pour l'export JSON

print("=" * 70)
print("  Expérience de comparaison des métaheuristiques pour le TSP")
print(f"  N_RUNS={N_RUNS}, MAX_EVALS={MAX_EVALS}")
print("=" * 70)

for inst_cfg in INSTANCES:
    inst_name = inst_cfg["name"]
    n = inst_cfg["n"]
    print(f"\n{'─'*60}")
    print(f"  Instance : {inst_name}  (n={n} villes)")
    print(f"{'─'*60}")

    cities, dist = ensure_instance(inst_cfg)
    raw_data[inst_name] = {}

    # Récupère les courbes de convergence pour une exécution représentative par algo
    convergence_curves = {}

    for algo in ALGORITHMS:
        costs  = []
        times  = []
        last_conv = None

        for run_id in range(N_RUNS):
            seed = inst_cfg["seed"] * 1000 + run_id * 7 + hash(algo) % 997
            cost, elapsed, conv = run_once(algo, n, dist, seed)
            costs.append(cost)
            times.append(elapsed)
            if run_id == 0 and conv is not None:
                last_conv = conv

        best   = min(costs)
        mean   = statistics.mean(costs)
        stdev  = statistics.stdev(costs) if len(costs) > 1 else 0.0
        t_mean = statistics.mean(times)

        row = {
            "instance":  inst_name,
            "n":         n,
            "algorithm": algo,
            "best":      round(best, 2),
            "mean":      round(mean, 2),
            "std":       round(stdev, 2),
            "time_mean": round(t_mean, 4),
        }
        all_results.append(row)
        raw_data[inst_name][algo] = {"costs": costs, "times": times}

        if last_conv is not None:
            convergence_curves[algo] = last_conv

        print(f"  {algo:<16} meilleur={best:>10.2f}  moyenne={mean:>10.2f}"
              f"  écart-type={stdev:>8.2f}  t={t_mean:.3f}s")

    # ── Tracer les courbes de convergence ──────────────────────
    if HAS_MPL and convergence_curves:
        fig, ax = plt.subplots(figsize=(10, 5))
        for algo, conv in convergence_curves.items():
            if conv and len(conv[0]) == 3:
                # Format SA / Tabu : (éval, actuel, meilleur)
                xs = [pt[0] for pt in conv]
                ys = [pt[2] for pt in conv]
            else:
                # Format MS / GRASP : (éval, meilleur)
                xs = [pt[0] for pt in conv]
                ys = [pt[1] for pt in conv]
            ax.plot(xs, ys, label=algo)

        ax.set_title(f"Courbes de convergence – {inst_name} (exécution 0)")
        ax.set_xlabel("Nombre d'évaluations de l'objectif")
        ax.set_ylabel("Meilleure longueur de tournée")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f"convergence_{inst_name}.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  [Graphique sauvegardé → {plot_path}]")


# ══════════════════════════════════════════════════════
#  SAUVEGARDE DES RÉSULTATS
# ══════════════════════════════════════════════════════

# Résumé CSV
csv_path = os.path.join(RESULTS_DIR, "summary.csv")
fieldnames = ["instance", "n", "algorithm", "best", "mean", "std", "time_mean"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)
print(f"\n[Résumé CSV sauvegardé → {csv_path}]")

# Données brutes JSON
json_path = os.path.join(RESULTS_DIR, "raw_data.json")
with open(json_path, "w") as f:
    json.dump(raw_data, f, indent=2)
print(f"[Données brutes JSON sauvegardées → {json_path}]")

# Affichage formaté du tableau final
print("\n" + "=" * 70)
print("  TABLEAU RÉCAPITULATIF FINAL")
print("=" * 70)
print(f"{'Instance':<20} {'Algorithme':<16} {'Meilleur':>10} {'Moyenne':>10} {'Écart-type':>10} {'Temps(s)':>8}")
print("─" * 70)
for row in all_results:
    print(f"{row['instance']:<20} {row['algorithm']:<16} "
          f"{row['best']:>10.2f} {row['mean']:>10.2f} "
          f"{row['std']:>10.2f} {row['time_mean']:>8.4f}")

print("\nTerminé.")