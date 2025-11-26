# 🚂 Panoptique Ferroviaire - Installation Sans Docker

Ce guide explique comment installer et exécuter l'application **sans Docker**, en mode autonome ou avec les services installés localement.

## 📋 Sommaire

- [Mode 1: Mode Autonome (Standalone)](#mode-1-mode-autonome-standalone) - **Recommandé pour démarrer**
- [Mode 2: Installation Complète Locale](#mode-2-installation-complète-locale)
- [Résolution des problèmes](#résolution-des-problèmes)

---

## Mode 1: Mode Autonome (Standalone)

Ce mode permet d'exécuter l'application sans Kafka ni PostgreSQL. Idéal pour:
- Tester les fonctionnalités de base
- Démonstrations
- Développement

### Prérequis

- **Python 3.12+**
- **pip** (gestionnaire de paquets Python)

### Étapes d'installation

```bash
# 1. Cloner le dépôt
git clone <repository-url>
cd PositionMetro

# 2. Créer un environnement virtuel
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# OU sur Windows:
# venv\Scripts\activate

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 4. Préparer les données statiques (optionnel mais recommandé)
python -m src.tools.prepare_data
```

### Exécution en mode autonome

```bash
# Exécuter la démonstration (fonctionne sans services externes)
python demo.py

# OU lancer le système en mode autonome
python main_standalone.py

# OU lancer la démo Rail-Lock
python demo_rail_lock.py
```

### Fonctionnalités disponibles en mode autonome

| Fonctionnalité | Disponible | Notes |
|----------------|------------|-------|
| Auto-découverte GTFS-RT | ✅ Oui | Via API transport.data.gouv.fr |
| Simulation physique | ✅ Oui | Équation de Davis complète |
| Filtre de Kalman | ✅ Oui | Estimation d'état |
| Moving Block (Cantonnement) | ✅ Oui | Prévention des collisions |
| Rail-Lock (v6.0) | ✅ Oui | Projection sur la voie |
| Holographic Positioning | ✅ Oui | Si stops.txt disponible |
| Streaming Kafka | ❌ Non | Requiert Kafka |
| Stockage PostGIS | ❌ Non | Requiert PostgreSQL |
| Kafka UI | ❌ Non | Requiert Kafka |

---

## Mode 2: Installation Complète Locale

Ce mode installe tous les services manuellement sur votre machine.

### Prérequis

- **Python 3.12+**
- **Java 11+** (pour Kafka et Zookeeper)
- **PostgreSQL 15+** avec PostGIS
- **Espace disque**: ~5 GB minimum

### Étape 1: Installation de PostgreSQL avec PostGIS

#### Sur Ubuntu/Debian

```bash
# Installer PostgreSQL et PostGIS
sudo apt-get update
sudo apt-get install -y postgresql-15 postgresql-15-postgis-3 postgresql-15-pgrouting

# Démarrer PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Créer la base de données
sudo -u postgres psql <<EOF
CREATE USER panoptique WITH PASSWORD 'panoptique_secure_2024';
CREATE DATABASE panoptique OWNER panoptique;
\c panoptique
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS pgrouting;
GRANT ALL PRIVILEGES ON DATABASE panoptique TO panoptique;
EOF

# Initialiser le schéma
PGPASSWORD=panoptique_secure_2024 psql -h localhost -U panoptique -d panoptique -f init-db/01-init.sql
```

#### Sur macOS (avec Homebrew)

```bash
# Installer PostgreSQL et PostGIS
brew install postgresql@15 postgis pgrouting

# Démarrer PostgreSQL
brew services start postgresql@15

# Créer la base de données
createuser panoptique
createdb -O panoptique panoptique
psql -d panoptique -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql -d panoptique -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"
psql -d panoptique -c "CREATE EXTENSION IF NOT EXISTS pgrouting;"

# Initialiser le schéma
psql -U panoptique -d panoptique -f init-db/01-init.sql
```

#### Sur Windows

1. Télécharger PostgreSQL depuis: https://www.postgresql.org/download/windows/
2. Installer avec l'option "PostGIS" dans Stack Builder
3. Créer la base de données via pgAdmin ou psql

### Étape 2: Installation d'Apache Kafka

#### Téléchargement et installation

```bash
# Créer un répertoire pour Kafka
mkdir -p ~/kafka && cd ~/kafka

# Télécharger Kafka (version 3.6.0)
curl -O https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0
```

#### Démarrer Zookeeper

```bash
# Terminal 1: Démarrer Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties
```

#### Démarrer Kafka

```bash
# Terminal 2: Démarrer Kafka
bin/kafka-server-start.sh config/server.properties
```

#### Créer le topic Kafka

```bash
# Terminal 3: Créer le topic raw_telemetry
bin/kafka-topics.sh --create \
    --topic raw_telemetry \
    --bootstrap-server localhost:9092 \
    --partitions 8 \
    --replication-factor 1
```

### Étape 3: Configuration de l'application

```bash
# Copier le fichier d'environnement
cp .env.example .env

# Éditer les variables (si nécessaire)
nano .env
```

Contenu du fichier `.env`:

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_RAW_TELEMETRY=raw_telemetry
KAFKA_GROUP_ID=neural_engine

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=panoptique
POSTGRES_USER=panoptique
POSTGRES_PASSWORD=panoptique_secure_2024

# Application
LOG_LEVEL=INFO
SIMULATION_RATE=1.0
HARVEST_INTERVAL=30.0
UVLOOP_ENABLED=true
MAX_CONCURRENT_HARVESTS=20
```

### Étape 4: Installation des dépendances Python

```bash
# Créer l'environnement virtuel
python3.12 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Préparer les données statiques
python -m src.tools.prepare_data
```

### Étape 5: Exécution du système complet

```bash
# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'application principale
python main.py
```

### Vérification du bon fonctionnement

1. **Vérifier PostgreSQL**:
```bash
psql -h localhost -U panoptique -d panoptique -c "SELECT COUNT(*) FROM railway.trains;"
```

2. **Vérifier Kafka**:
```bash
# Lister les topics
bin/kafka-topics.sh --list --bootstrap-server localhost:9092

# Consommer les messages
bin/kafka-console-consumer.sh --topic raw_telemetry --bootstrap-server localhost:9092 --from-beginning
```

3. **Vérifier l'application**:
```bash
# Logs dans logs/panoptique.log
tail -f logs/panoptique.log
```

---

## Résolution des problèmes

### Erreur: "uvloop not available"

```bash
# uvloop n'est pas disponible sur Windows
# L'application utilise asyncio standard automatiquement
```

### Erreur: "Kafka connection refused"

```bash
# Vérifier que Zookeeper est démarré
ps aux | grep zookeeper

# Vérifier que Kafka est démarré
ps aux | grep kafka

# Vérifier les ports
netstat -an | grep 9092
netstat -an | grep 2181
```

### Erreur: "PostgreSQL connection failed"

```bash
# Vérifier que PostgreSQL est démarré
sudo systemctl status postgresql

# Vérifier la connexion
psql -h localhost -U panoptique -d panoptique -c "SELECT 1;"
```

### Erreur: "gtfs-realtime-bindings import error"

```bash
# Réinstaller les dépendances protobuf
pip uninstall protobuf gtfs-realtime-bindings
pip install protobuf>=4.25.0 gtfs-realtime-bindings>=1.0.0
```

### Erreur: "No stops.txt or topology.json"

```bash
# Exécuter le script de préparation des données
python -m src.tools.prepare_data

# Vérifier les fichiers générés
ls -la data/
```

---

## Scripts de commodité

### Script de démarrage complet (Linux/Mac)

Créer un fichier `start_local.sh`:

```bash
#!/bin/bash

echo "🚂 Démarrage de Panoptique Ferroviaire (mode local)"

# Vérifier PostgreSQL
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo "❌ PostgreSQL n'est pas démarré"
    exit 1
fi
echo "✓ PostgreSQL OK"

# Vérifier Kafka (optionnel)
if nc -z localhost 9092 > /dev/null 2>&1; then
    echo "✓ Kafka OK"
else
    echo "⚠ Kafka non disponible - mode dégradé"
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'application
python main.py
```

### Script d'arrêt (Linux/Mac)

Créer un fichier `stop_local.sh`:

```bash
#!/bin/bash

echo "🛑 Arrêt de Panoptique Ferroviaire"

# Arrêter l'application Python
pkill -f "python main.py" 2>/dev/null

echo "✓ Application arrêtée"
```

---

## Comparaison: Docker vs Local

| Aspect | Docker | Local |
|--------|--------|-------|
| Installation | Simple (`docker-compose up`) | Complexe (plusieurs services) |
| Isolation | Complète | Partagée |
| Ressources | Plus de mémoire | Moins de mémoire |
| Debugging | Plus difficile | Plus facile |
| Production | Recommandé | Non recommandé |
| Développement | OK | Recommandé |

---

## Support

Pour toute question ou problème:
1. Vérifier les logs: `logs/panoptique.log`
2. Consulter la documentation: `docs/`
3. Lancer les tests: `pytest tests/ -v`

---

**Version**: 6.0.0  
**Mode**: Installation Sans Docker
