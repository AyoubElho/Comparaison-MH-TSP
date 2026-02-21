"""
generate_instances.py
Génère des instances aléatoires du problème du voyageur de commerce (TSP)
et les sauvegarde au format JSON.
"""

import json
import random
import math
import os


def generate_random_instance(n, seed=None, bounds=(0, 100)):
    """
    Génère une instance TSP aléatoire contenant n villes.

    Paramètres
    ----------
    n      : nombre de villes
    seed   : graine aléatoire pour reproductibilité
    bounds : intervalle des coordonnées (xmin, xmax)

    Retourne
    -------
    Liste de tuples (x, y)
    """
    if seed is not None:
        random.seed(seed)

    cities = [
        (
            round(random.uniform(*bounds), 2),
            round(random.uniform(*bounds), 2)
        )
        for _ in range(n)
    ]

    return cities


def save_instance(cities, filepath):
    """
    Sauvegarde une instance TSP dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump({"cities": cities, "n": len(cities)}, f, indent=2)

    print(f"Instance sauvegardée : {len(cities)} villes → {filepath}")


def load_instance(filepath):
    """
    Charge une instance TSP depuis un fichier JSON.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data["cities"]


def euclidean_distance(c1, c2):
    """
    Calcule la distance euclidienne entre deux villes.
    """
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def build_distance_matrix(cities):
    """
    Construit la matrice complète des distances
    à partir des coordonnées des villes.
    """
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclidean_distance(cities[i], cities[j])

    return dist


if __name__ == "__main__":

    # Liste des instances à générer : (nom_fichier, nombre_villes, graine)
    instances = [
        ("instance_A_20.json", 20, 42),
        ("instance_B_50.json", 50, 123),
        ("instance_C_80.json", 80, 999),
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for filename, n, seed in instances:
        cities = generate_random_instance(n, seed=seed)
        save_instance(cities, os.path.join(base_dir, filename))