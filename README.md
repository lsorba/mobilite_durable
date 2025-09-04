# 🚌 Projet MDM - Mobilité Durable en Montagne ⛰️

En collaboration avec :
* DataForGood Grenoble : https://data-for-good-grenoble.github.io/
* CampToCamp : https://www.camptocamp.org/
* ProtectOurWinters : https://protectourwinters.fr/

Spécial remerciement :
* https://transport.data.gouv.fr

## 📊 Exploration des données

Les notebooks d’analyse de données sont regroupés dans le dossier `src/notebooks`.

⚠️ **Attention à la gestion de version des notebooks Jupyter** : en raison des risques fréquents de **conflits lors des modifications**, il est recommandé de **créer un nouveau notebook** plutôt que de modifier un notebook existant que vous n’avez pas vous-même créé. Une fusion des différentes versions sera effectuée ultérieurement si nécessaire.

### 🗂️ Nomenclature des notebooks

Pour faciliter la collaboration et le suivi des modifications, les notebooks doivent être nommés selon la convention suivante :

```
<date>_<auteur>_<feature>.ipynb
```
- `date` : au format `YYYY-MM-DD` (exemple `2025-06-19`)
- `auteur` : prénom ou identifiant Git (exemple `AReboud`)
- `feature` : description courte (exemple `EPSG`)

```
2025-06-19_AReboud_EPSG.ipynb
```

## Installation

### Dépendances

[Python](https://www.python.org/downloads/) 3.13\
[uv](https://docs.astral.sh/uv/getting-started/installation/), le gestionnaire de paquet

## Générer l'environnement virtuel
```sh
uv sync
uv pip install ".[dev,test]"
```

## [Activer l'environnement virtuel](https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment)

macOS and Linux:

```sh
source .venv/bin/activate
```

windows:

```sh
.venv\Scripts\activate
```

## Exécuter un script

```sh
python main.py path/to/script.py
```

ou si on n'a pas activé l'environnement

```sh
uv run main.py path/to/script.py
```

## Activation du pre-commit

```sh
pre-commit install
```

## Jouer avec Jupyter Lab

```sh
.venv/bin/jupyter lab
```


## Doc utile

- Format GTFS : https://gtfs.org/documentation/schedule/reference/
- gtfs-kit : https://github.com/mrcagney/gtfs_kit
- …


## Repère géospatial

Le repère géospatial par défaut du projet est
> EPSG:3857

C'est lui que l'on doit trouver dans les différents jeux de données.
Celui-ci est optimisé pour la visualisation des données sur écran.

Lors des calculs géospatiaux, ce repère est source d'erreur, dans ce cas, on utilise le 
> EPSG:4326
