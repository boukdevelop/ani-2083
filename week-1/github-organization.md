# 🌿 Organisation GitHub — Projet Reconnaissance Faciale

> Règles de collaboration pour l'équipe (5 étudiants). Objectif : éviter les conflits et garder
> un historique propre pendant les 6 jours du projet.

---

## 1. Convention de nommage des branches

Format : `type/module-ou-tache`

| Type    | Usage                                              | Exemple               |
| :------ | :------------------------------------------------- | :-------------------- |
| `main`  | Branche stable, **protégée** — code qui fonctionne | `main`                |
| `dev`   | Branche d'intégration (optionnelle)                | `dev`                 |
| `feat/` | Nouvelle fonctionnalité                            | `feat/face-encoder`   |
| `fix/`  | Correction de bug                                  | `fix/tolerance-seuil` |
| `docs/` | Documentation                                      | `docs/readme`         |
| `test/` | Tests                                              | `test/unknown-cases`  |

**Règle** : une branche = **un étudiant = un module**. Nom en minuscules, séparateurs `-`.

```
feat/capture-dataset      # Étudiant 1
feat/face-encoder         # Étudiant 2
feat/face-matcher         # Étudiant 3
feat/camera-stream        # Étudiant 4
feat/gui-integration      # Étudiant 5
```

---

## 2. Règles de collaboration

1. **Jamais de commit direct sur `main`** — tout passe par une Pull Request (PR).
2. **Une branche par tâche** — pas de travail croisé sur la branche d'un autre.
3. **Pull avant de pousser** : `git pull origin main` pour rester à jour.
4. **Commits fréquents et petits** — un commit = une action logique.
5. **PR obligatoire** pour fusionner dans `main`, avec au moins **1 relecture** (le chef de projet).
6. **Ne jamais committer de données biométriques** : images de `data/` et `encoded_faces.pkl`
   restent **locaux** (voir `.gitignore`).
7. **Conflits** : prévenir le chef de projet avant de forcer quoi que ce soit (`--force` interdit sur `main`).

---

## 3. Convention de messages de commit

Format : `type(module): description courte`

```
feat(capture): ajoute la capture auto de 30 images
fix(matcher): corrige le calcul du score de confiance
docs(readme): ajoute la section installation
test(encoder): teste l'encodage d'images sans visage
refactor(gui): simplifie la boucle after()
```

| Type       | Sens                                       |
| :--------- | :----------------------------------------- |
| `feat`     | Nouvelle fonctionnalité                    |
| `fix`      | Correction de bug                          |
| `docs`     | Documentation                              |
| `test`     | Ajout/modification de tests                |
| `refactor` | Réécriture sans changement de comportement |
| `chore`    | Config, dépendances, nettoyage             |

---

## 4. Workflow type (résumé)

```powershell
git checkout main
git pull origin main
git checkout -b feat/face-encoder        # 1. créer sa branche
# ... coder son module ...
git add face_encoder.py
git commit -m "feat(encoder): génère encoded_faces.pkl"
git push origin feat/face-encoder         # 2. pousser
# 3. ouvrir une Pull Request vers main → relecture → fusion
```

---

## 5. Fichier `.gitignore` (racine du projet)

```gitignore
# Environnement
.venv/
__pycache__/
*.pyc

# Données biométriques (RGPD) — NE JAMAIS VERSIONNER
data/known_faces/
data/unknown_tests/
encoded_faces.pkl

# Divers
.vscode/
.idea/
*.zip
```

---

## 6. Bonnes pratiques

- **Puller souvent** : au moins une fois par jour, avant de commencer à coder.
- **Tester avant de pousser** : vérifier que son module s'exécute sans erreur.
- **PR courte** : limiter à un module / une fonctionnalité pour faciliter la relecture.
- **Communiquer** : prévenir l'équipe sur le canal de discussion avant de toucher un fichier partagé
  (`main.py`, `requirements.txt`).
- **Protéger `main`** : activer la protection de branche dans GitHub (Settings → Branches).
