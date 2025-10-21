# 🐍 Q-Learning Snake Trainer

## 🚀 Overview
Ce projet met en œuvre un **agent d’apprentissage par renforcement (Q-Learning)** dans un environnement simplifié de type *Snake*.  
L’agent apprend à se déplacer sur une carte 15×15 pour atteindre une position cible (la pomme 🍎) sans sortir des limites de la carte.  
Une visualisation graphique en **Pygame** permet d’observer les déplacements de l’agent après l’entraînement.

---

## 📁 Structure du projet

```

├── main.py                # Script principal : logique Q-learning + visualisation
├── data/
│   ├── sprites/
│   │   ├── snake.png      # Sprite du serpent (agent)
│   │   └── appel.png      # Sprite de la pomme
│   └── map/
│       └── map_snake.tmx  # Carte du jeu (format Tiled Map Editor)
├── README.md              # Ce fichier

````

---

## ⚙️ Fonctionnement général

### 🧠 1. Entraînement (Q-Learning)
L’agent, appelé `SubjectOfTraining`, apprend à atteindre la cible `(5, 5)` sur une grille de taille 15×15 à l’aide du **Q-learning**.

#### Paramètres d’apprentissage :
| Paramètre | Symbole | Valeur par défaut | Rôle |
|------------|----------|------------------|------|
| Taux d’apprentissage | α | 0.5 | Pondère la mise à jour du Q-table |
| Facteur de récompense future | γ | 0.95 | Importance des récompenses futures |
| Taux d’exploration | ε | 0.1 (décroissant) | Probabilité de choisir une action aléatoire |
| Nombre d’épisodes | N | 1000 (par défaut) | Nombre de parties d’entraînement |

#### Récompenses :
- 🏁 **Atteindre la pomme** : `+10`
- 🚫 **Sortir de la carte** : `-10`
- 🕒 **Chaque pas** : `-0.05`

L’objectif est d’apprendre une politique (table Q) qui maximise la récompense totale.

---

### 🎮 2. Visualisation (Pygame)
Après l’entraînement, le script peut afficher une **animation** d’un épisode particulier pour visualiser la trajectoire de l’agent.

L’animation affiche :
- La carte (`map_snake.tmx`)
- Le serpent (agent)
- La pomme (objectif fixe à `(5, 5)`)

---

## 🧩 Classes principales

| Classe | Rôle |
|--------|------|
| **`SubjectOfTraining`** | Définit l’agent (position, mouvement, réinitialisation) |
| **`MasterOfTraining`** | Implémente le Q-learning (choix d’action, mise à jour de la Q-table, calcul des récompenses) |
| **`EnvironnementOfTrainning`** | Définit les dimensions de la grille |
| **`RunProgramTraining`** | Gère la boucle d’entraînement complète |
| **`PrintOneEpisode`** | Affiche un épisode en Pygame |

---

## 🚀 Lancer le projet

### 1. Installer les dépendances
Assure-toi d’avoir Python ≥ 3.8 puis installe les librairies nécessaires :

```bash
pip install pygame pytmx numpy
````

### 2. Lancer l’entraînement

Depuis le dossier du projet :

```bash
python main.py
```

### 3. Observer le résultat

À la fin de l’entraînement, Pygame s’ouvre et affiche le déplacement du serpent pour un épisode donné (par défaut : le 999ᵉ).

---

## 🧠 Concepts utilisés

* **Q-Learning** : apprentissage tabulaire basé sur l’équation de Bellman
  ( Q(s,a) = (1 - \alpha) Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a')] )
* **Exploration/Exploitation** : équilibre entre tester de nouvelles actions (ε-greedy) et exploiter ce que l’on a déjà appris.
* **Récompense différée** : l’agent apprend à anticiper les récompenses à long terme.

---

## 📊 Améliorations possibles

* ✅ Ajout de **graphique matplotlib** pour suivre l’évolution des récompenses par épisode
* ✅ Sauvegarde/chargement de la Q-table (`numpy.save` / `numpy.load`)
* ✅ Ajout de **collision avec soi-même** (snake complet)
* ✅ Variation de la position de la pomme pour un apprentissage plus général
* ✅ Passage à une version **Deep Q-Learning (DQN)** avec un réseau de neurones (PyTorch ou TensorFlow)

---

## 💡 Exemple de résultat attendu (conceptuellement)

Après plusieurs milliers d’épisodes :

* Le serpent apprend à se diriger vers `(5, 5)` rapidement.
* Les épisodes terminent plus vite et avec des récompenses plus élevées.
* La Q-table contient les meilleures actions pour chaque position.

---

## 🧑‍💻 Auteur

**Projet de démonstration de Q-Learning avec Pygame**
Développé pour l’étude de l’apprentissage par renforcement sur des environnements simples en grille.

---

## 🌟 License

Projet libre à usage éducatif.
© 2025 — Tu peux le modifier et le redistribuer librement.


