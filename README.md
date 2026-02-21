# Projet Métaheuristiques TSP
**Université Hassan II – ENSET Mohammedia | Masters SDIA** *UE : Optimisation / Métaheuristiques — Prof. MESTARI — A.U. 25-26*

---

## 📁 Structure du Projet

    tsp_project/
    ├── data/
    │   ├── generate_instances.py   # Générateur d'instances & utilitaires
    │   ├── instance_A_20.json      # Auto-généré : 20 villes
    │   ├── instance_B_50.json      # Auto-généré : 50 villes
    │   └── instance_C_80.json      # Auto-généré : 80 villes
    ├── algorithms.py               # Toutes les métaheuristiques
    ├── experiment.py               # Lanceur d'expériences & exportateur de résultats
    ├── results/
    │   ├── summary.csv             # Tableau récapitulatif (meilleur/moyenne/écart-type/temps)
    │   ├── raw_data.json           # Coûts bruts et temps par exécution
    │   └── convergence_*.png       # Graphiques des courbes de convergence
    └── README.md

---

## 🧠 Algorithmes Implémentés

| Nom | Description |
|---|---|
| `HC_Best` | Hill Climbing (Recherche locale) – Meilleure amélioration (échange) |
| `HC_First` | Hill Climbing – Première amélioration (échange) |
| `HC_Best_2opt` | Hill Climbing – Meilleure amélioration (2-opt) |
| `HC_First_2opt` | Hill Climbing – Première amélioration (2-opt) |
| `MS_HC` | Hill Climbing Multi-Départ (30 redémarrages) |
| `SA` | Recuit Simulé (échange, refroidissement géométrique) |
| `SA_2opt` | Recuit Simulé (mouvements 2-opt) |
| `Tabu` | Recherche Tabou *(bonus)* |
| `GRASP` | GRASP – Construction gloutonne aléatoire + HC *(bonus)* |

---

## ⚙️ Prérequis

    pip install matplotlib   # optionnel, pour les graphiques

Aucune autre bibliothèque externe n'est requise — Python 3 pur.

---

## 🚀 Comment Exécuter

### Étape 1 – Générer les instances (optionnel, fait automatiquement par experiment.py)

    cd tsp_project/
    python data/generate_instances.py

### Étape 2 – Lancer l'expérience complète

    python experiment.py


Ceci va :
1. Générer les trois instances du TSP si elles ne sont pas déjà présentes.
2. Exécuter les 9 algorithmes × 3 instances × 30 exécutions indépendantes.
3. Afficher un tableau récapitulatif dans la console.
4. Sauvegarder `results/summary.csv` et `results/raw_data.json`.
5. Sauvegarder les graphiques des courbes de convergence dans `results/convergence_*.png`.

---

## 📊 Exemple de Sortie (console)

    Instance             Algorithme       Meilleur   Moyenne   Ecart-type  Temps(s)
    ─────────────────────────────────────────────────────────────────────────────────
    Instance_A_20        HC_Best          450.12     470.30    15.20       0.0012
    Instance_A_20        SA               432.88     445.10    12.50       0.3400
    ...

---

## 🔧 Configuration (experiment.py)

Vous pouvez facilement ajuster les paramètres de l'expérience en haut du fichier `experiment.py` :

    N_RUNS      = 30        # nombre d'exécutions indépendantes par algo et par instance
    MAX_EVALS   = 200_000   # budget d'évaluations par exécution
    
    SA_T0       = 1000.0    # température initiale du Recuit Simulé (SA)
    SA_ALPHA    = 0.995     # taux de refroidissement du Recuit Simulé
    SA_TMIN     = 1e-3      # température d'arrêt du Recuit Simulé
    
    MS_STARTS   = 30        # nombre de redémarrages pour le HC Multi-Départ
    TABU_TENURE = 10        # taille de la liste tabou (tenure)
    GRASP_ITERATIONS = 30   # itérations de construction GRASP
    GRASP_ALPHA = 0.3       # seuil LCR (0=glouton, 1=aléatoire)

---

## 📐 Modèle du TSP

- **Villes** : coordonnées (x, y) ∈ [0, 100]²
- **Distance** : Euclidienne d(i, j) = √((xᵢ - xⱼ)² + (yᵢ - yⱼ)²)
- **Solution** : permutation π de {0, …, n−1}
- **Objectif** (minimiser) : Σ d(π_k, π_{k+1 mod n})
- **Voisinages** : échange (swap) de deux villes / inversion de segment (2-opt)

---

## 📝 Notes

- Tous les algorithmes partagent le même budget d'évaluations (`MAX_EVALS`) pour permettre une comparaison équitable.
- Chaque exécution utilise une graine aléatoire (seed) reproductible différente, dérivée de la graine de l'instance et de l'index de l'exécution.
- Les courbes de convergence sont sauvegardées pour l'exécution 0 de chaque algorithme.
