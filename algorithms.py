"""
algorithms.py
Implémentation de métaheuristiques pour le Problème du Voyageur de Commerce (TSP).

Algorithmes inclus :
  - Hill Climbing (Recherche Locale) : Meilleure Amélioration & Première Amélioration
  - Hill Climbing Multi-Départ
  - Recuit Simulé (SA)
  - Recherche Tabou (bonus)
  - GRASP (bonus)
"""

import random
import math
import time
from copy import deepcopy


# ─────────────────────────────────────────────
#  Utilitaires TSP
# ─────────────────────────────────────────────

def tour_length(tour, dist):
    """Calcule la longueur totale d'une tournée fermée."""
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def random_tour(n):
    """Génère une permutation aléatoire de n villes."""
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def greedy_tour(n, dist):
    """
    Construit une tournée gloutonne en utilisant l'heuristique du plus proche voisin,
    en partant d'une ville aléatoire.
    """
    start = random.randint(0, n - 1)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        current = tour[-1]
        best_next = None
        best_dist = float('inf')
        for j in range(n):
            if not visited[j] and dist[current][j] < best_dist:
                best_dist = dist[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True
    return tour


# ─────────────────────────────────────────────
#  Opérateurs de voisinage
# ─────────────────────────────────────────────

def swap_neighbors(tour):
    """
    Générateur : produit tous les voisins obtenus en échangeant
    deux positions i < j dans la tournée (voisinage par échange/swap).
    Produit (new_tour, i, j).
    """
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = tour[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor, i, j


def two_opt_neighbors(tour):
    """
    Générateur : produit tous les voisins obtenus en inversant
    un segment [i+1 … j] (voisinage 2-opt).
    Produit (new_tour, i, j).
    """
    n = len(tour)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
            yield neighbor, i, j


def swap_delta(tour, dist, i, j):
    """
    Calcule le delta de coût pour un échange (swap) des positions i et j SANS construire la tournée complète.
    Gère les cas limites où i et j sont adjacents.
    """
    n = len(tour)
    # villes impliquées
    a, b = tour[i], tour[j]
    # voisins dans la tournée actuelle
    prev_i = tour[(i - 1) % n]
    next_i = tour[(i + 1) % n]
    prev_j = tour[(j - 1) % n]
    next_j = tour[(j + 1) % n]

    if (i + 1) % n == j:
        # adjacents : i -> j
        old = dist[prev_i][a] + dist[a][b] + dist[b][next_j]
        new = dist[prev_i][b] + dist[b][a] + dist[a][next_j]
    elif (j + 1) % n == i:
        # adjacents : j -> i
        old = dist[prev_j][b] + dist[b][a] + dist[a][next_i]
        new = dist[prev_j][a] + dist[a][b] + dist[b][next_i]
    else:
        old = (dist[prev_i][a] + dist[a][next_i] +
               dist[prev_j][b] + dist[b][next_j])
        new = (dist[prev_i][b] + dist[b][next_i] +
               dist[prev_j][a] + dist[a][next_j])

    return new - old


# ─────────────────────────────────────────────
#  1. Hill Climbing – Meilleure Amélioration
# ─────────────────────────────────────────────

def hill_climbing_best(initial_tour, dist, max_evals=None, use_2opt=False):
    """
    Hill Climbing avec stratégie de meilleure amélioration.

    Paramètres
    ----------
    initial_tour : list[int]
    dist         : Matrice de distance 2D
    max_evals    : Nombre maximum d'évaluations de la fonction objectif (None = illimité)
    use_2opt     : Si True, utilise le voisinage 2-opt, sinon l'échange (swap)

    Retours
    -------
    best_tour, best_cost, n_evals
    """
    neighbor_fn = two_opt_neighbors if use_2opt else swap_neighbors

    tour = initial_tour[:]
    cost = tour_length(tour, dist)
    n_evals = 1
    improved = True

    while improved:
        improved = False
        best_neighbor = None
        best_cost = cost

        for neighbor, i, j in neighbor_fn(tour):
            if max_evals and n_evals >= max_evals:
                return tour, cost, n_evals
            neighbor_cost = tour_length(neighbor, dist)
            n_evals += 1
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                best_neighbor = neighbor[:]

        if best_neighbor is not None:
            tour = best_neighbor
            cost = best_cost
            improved = True

    return tour, cost, n_evals


# ─────────────────────────────────────────────
#  2. Hill Climbing – Première Amélioration
# ─────────────────────────────────────────────

def hill_climbing_first(initial_tour, dist, max_evals=None, use_2opt=False):
    """
    Hill Climbing avec stratégie de première amélioration.

    Retours
    -------
    best_tour, best_cost, n_evals
    """
    neighbor_fn = two_opt_neighbors if use_2opt else swap_neighbors

    tour = initial_tour[:]
    cost = tour_length(tour, dist)
    n_evals = 1
    improved = True

    while improved:
        improved = False
        for neighbor, i, j in neighbor_fn(tour):
            if max_evals and n_evals >= max_evals:
                return tour, cost, n_evals
            neighbor_cost = tour_length(neighbor, dist)
            n_evals += 1
            if neighbor_cost < cost:
                tour = neighbor[:]
                cost = neighbor_cost
                improved = True
                break  # redémarre immédiatement à partir de la nouvelle position

    return tour, cost, n_evals


# ─────────────────────────────────────────────
#  3. Hill Climbing Multi-Départ
# ─────────────────────────────────────────────

def multi_start_hill_climbing(n, dist, n_starts=30, hc_variant='best',
                               use_2opt=False, max_evals=None):
    """
    Hill Climbing Multi-Départ.

    Paramètres
    ----------
    n          : nombre de villes
    dist       : matrice de distance
    n_starts   : nombre de redémarrages aléatoires
    hc_variant : 'best' (meilleure) ou 'first' (première)
    use_2opt   : utiliser le voisinage 2-opt
    max_evals  : budget total d'évaluations (partagé entre les redémarrages)

    Retours
    -------
    best_tour, best_cost, total_evals, convergence
    """
    hc_fn = hill_climbing_best if hc_variant == 'best' else hill_climbing_first

    best_tour = None
    best_cost = float('inf')
    total_evals = 0
    convergence = []  # (evals_jusqu_ici, meilleur_cout_jusqu_ici)

    for _ in range(n_starts):
        if max_evals and total_evals >= max_evals:
            break
        remaining = (max_evals - total_evals) if max_evals else None
        init = random_tour(n)
        tour, cost, evals = hc_fn(init, dist, max_evals=remaining, use_2opt=use_2opt)
        total_evals += evals
        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]
        convergence.append((total_evals, best_cost))

    return best_tour, best_cost, total_evals, convergence


# ─────────────────────────────────────────────
#  4. Recuit Simulé
# ─────────────────────────────────────────────

def simulated_annealing(initial_tour, dist,
                         T0=1000.0, alpha=0.995, T_min=1e-3,
                         max_evals=100_000, use_2opt=False):
    """
    Recuit Simulé pour le TSP.

    Schéma de refroidissement : T_{k+1} = alpha * T_k

    Paramètres
    ----------
    initial_tour : permutation de départ
    dist         : matrice de distance
    T0           : température initiale
    alpha        : taux de refroidissement (0 < alpha < 1)
    T_min        : température d'arrêt
    max_evals    : budget maximum d'évaluations
    use_2opt     : utiliser les mouvements 2-opt (sinon l'échange)

    Retours
    -------
    best_tour, best_cost, n_evals, convergence
    """
    n = len(initial_tour)
    tour = initial_tour[:]
    cost = tour_length(tour, dist)

    best_tour = tour[:]
    best_cost = cost

    T = T0
    n_evals = 1
    convergence = []  # (num_eval, cout_actuel, meilleur_cout)
    convergence.append((0, cost, best_cost))

    while T > T_min and n_evals < max_evals:
        # Génère un voisin aléatoire
        if use_2opt:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            neighbor = tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]
        else:
            i, j = sorted(random.sample(range(n), 2))
            neighbor = tour[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        neighbor_cost = tour_length(neighbor, dist)
        n_evals += 1

        delta = neighbor_cost - cost

        # Critère d'acceptation
        if delta <= 0 or random.random() < math.exp(-delta / T):
            tour = neighbor
            cost = neighbor_cost
            if cost < best_cost:
                best_cost = cost
                best_tour = tour[:]

        T *= alpha

        if n_evals % 500 == 0:
            convergence.append((n_evals, cost, best_cost))

    convergence.append((n_evals, cost, best_cost))
    return best_tour, best_cost, n_evals, convergence


# ─────────────────────────────────────────────
#  5. Recherche Tabou (bonus)
# ─────────────────────────────────────────────

def tabu_search(initial_tour, dist, tabu_tenure=10,
                max_evals=50_000, use_2opt=False):
    """
    Recherche Tabou pour le TSP utilisant le voisinage par échange (ou 2-opt).
    La liste tabou interdit de réappliquer le même mouvement (i, j) pendant `tabu_tenure` itérations.

    Retours
    -------
    best_tour, best_cost, n_evals, convergence
    """
    neighbor_fn = two_opt_neighbors if use_2opt else swap_neighbors

    tour = initial_tour[:]
    cost = tour_length(tour, dist)
    best_tour = tour[:]
    best_cost = cost

    tabu_list = {}  # mouvement -> itération où il redevient libre
    iteration = 0
    n_evals = 1
    convergence = []

    while n_evals < max_evals:
        iteration += 1
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None

        for neighbor, i, j in neighbor_fn(tour):
            if n_evals >= max_evals:
                break
            neighbor_cost = tour_length(neighbor, dist)
            n_evals += 1
            move = (min(i, j), max(i, j))

            is_tabu = tabu_list.get(move, 0) >= iteration
            # Aspiration : accepter si meilleur que le meilleur global
            if (not is_tabu or neighbor_cost < best_cost):
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor_cost = neighbor_cost
                    best_neighbor = neighbor[:]
                    best_move = move

        if best_neighbor is None:
            break

        # Appliquer le mouvement
        tour = best_neighbor
        cost = best_neighbor_cost
        tabu_list[best_move] = iteration + tabu_tenure

        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]

        convergence.append((n_evals, cost, best_cost))

    return best_tour, best_cost, n_evals, convergence


# ─────────────────────────────────────────────
#  6. GRASP (bonus)
# ─────────────────────────────────────────────

def _grasp_construction(n, dist, alpha_rcl=0.3):
    """
    Procédure de Recherche Adaptative Aléatoire Gloutonne (GRASP) – phase de construction.
    Construit une tournée en utilisant une Liste Restreinte de Candidats (RCL).

    alpha_rcl : 0 → purement glouton, 1 → purement aléatoire
    """
    unvisited = set(range(n))
    start = random.randint(0, n - 1)
    tour = [start]
    unvisited.remove(start)

    while unvisited:
        current = tour[-1]
        costs = sorted((dist[current][j], j) for j in unvisited)
        c_min, c_max = costs[0][0], costs[-1][0]
        threshold = c_min + alpha_rcl * (c_max - c_min)
        rcl = [j for d, j in costs if d <= threshold]
        chosen = random.choice(rcl)
        tour.append(chosen)
        unvisited.remove(chosen)

    return tour


def grasp(n, dist, n_iterations=30, alpha_rcl=0.3,
          hc_variant='best', use_2opt=False, max_evals=None):
    """
    GRASP : construction aléatoire gloutonne répétée + recherche locale.

    Retours
    -------
    best_tour, best_cost, total_evals, convergence
    """
    hc_fn = hill_climbing_best if hc_variant == 'best' else hill_climbing_first

    best_tour = None
    best_cost = float('inf')
    total_evals = 0
    convergence = []

    for _ in range(n_iterations):
        if max_evals and total_evals >= max_evals:
            break
        remaining = (max_evals - total_evals) if max_evals else None

        init = _grasp_construction(n, dist, alpha_rcl=alpha_rcl)
        tour, cost, evals = hc_fn(init, dist, max_evals=remaining, use_2opt=use_2opt)
        total_evals += evals

        if cost < best_cost:
            best_cost = cost
            best_tour = tour[:]
        convergence.append((total_evals, best_cost))

    return best_tour, best_cost, total_evals, convergence