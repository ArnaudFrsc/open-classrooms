# Kit P9 — Pipeline Big Data Fruits sur AWS

## 📁 Fichiers du kit

| Fichier | Rôle | Où il s'exécute |
|---|---|---|
| `notebook_aws_studio.ipynb` | **Notebook interactif** (kernel PySpark) | EMR Studio, dans ton navigateur |
| `p9_pipeline.py` | **Script de production** (run batch reproductible) | `spark-submit` sur EMR |
| `bootstrap.sh` | Installe TF/Pillow/etc sur les nœuds | EMR (auto, au démarrage) |
| `launch_emr_interactive.sh` | Crée un cluster EMR pour notebook | Ta machine (Git Bash / WSL / Linux) |
| `launch_emr.sh` | Crée un cluster + lance le job en batch | Ta machine |

**Tu utilises les deux** : notebook pour explorer, script pour le run de prod final. Les deux validés.

---

## 🇪🇺 Conformité RGPD (rappel)

| Exigence | Implémentation |
|---|---|
| Stockage en zone UE | Bucket S3 créé en `eu-west-3` (Paris) |
| Calcul en zone UE | Cluster EMR provisionné en `eu-west-3` |
| Chiffrement au repos | SSE-S3 (par défaut sur tout bucket S3) |
| Chiffrement en transit | HTTPS pour S3, communications inter-nœuds dans le VPC |
| Pas d'API hors UE | Aucun service AWS d'autre région appelé |

⚠️ **Vérifie systématiquement la région** dans la console (en haut à droite). C'est le piège le plus fréquent.

---

## 🚀 Parcours étape par étape (depuis zéro)

### Étape 1 — Setup AWS local

Sur ta machine Windows :
```powershell
# Installer AWS CLI : https://aws.amazon.com/cli/
aws configure
# Renseigne : Access Key ID, Secret Access Key, Region (eu-west-3), Format (json)
```

Pour créer les Access Keys : Console AWS → IAM → Users → ton user → Security credentials → Create access key.

### Étape 2 — Créer le bucket S3

```bash
aws s3 mb s3://mon-bucket-p9-tonprenom --region eu-west-3
```

Ou via la console : S3 → Create bucket → **Region : Europe (Paris) eu-west-3**.

### Étape 3 — Uploader les données et le code

```bash
aws s3 cp bootstrap.sh    s3://mon-bucket-p9-tonprenom/code/
aws s3 cp p9_pipeline.py  s3://mon-bucket-p9-tonprenom/code/

aws s3 sync /chemin/local/fruits-360/Test s3://mon-bucket-p9-tonprenom/data/fruits-360/Test
```

L'upload du dataset peut prendre 10-20 min. Lance-le et passe à la suite en parallèle.

### Étape 4 — Rôles IAM par défaut

```bash
aws emr create-default-roles
```

### Étape 5 — Key pair EC2

```bash
aws ec2 create-key-pair --key-name p9-keypair --region eu-west-3 \
    --query 'KeyMaterial' --output text > p9-keypair.pem
```

### Étape 6 — Subnet ID

```bash
aws ec2 describe-subnets --region eu-west-3 \
    --query 'Subnets[?MapPublicIpOnLaunch==`true`] | [0].SubnetId' --output text
```

### Étape 7 — Lancer le cluster (mode notebook)

Édite `launch_emr_interactive.sh` (5 lignes en haut), puis :
```bash
chmod +x launch_emr_interactive.sh
./launch_emr_interactive.sh
```

Le cluster met **5-8 min** à être prêt. État final : **WAITING**.

### Étape 8 — Créer EMR Studio (une fois)

Console AWS → EMR → EMR Studio → Create Studio :
- Setup : **Custom**
- Authentication : **IAM**
- Service role : **Create new role**
- Workspace storage : `s3://mon-bucket-p9-tonprenom/studio-workspace/`
- VPC + Subnet : même que le cluster

### Étape 9 — Workspace + cluster

EMR Studio → Create Workspace → **Attach to an EMR cluster on EC2** → ton cluster.

### Étape 10 — Le notebook

1. Upload `notebook_aws_studio.ipynb` dans le Workspace
2. Ouvre-le, vérifie kernel = **PySpark**
3. Exécute la cellule 0 (`%%configure`) en premier
4. Modifie `S3_BUCKET` (cellule 2) avec ton vrai nom de bucket
5. Run cellule par cellule

### Étape 11 — Terminer le cluster

**Très important :**
```bash
aws emr terminate-clusters --cluster-ids $(cat .last_cluster_id) --region eu-west-3
```

L'`IdleTimeout=3600` est un filet de sécurité, pas une excuse pour oublier.

---

## 🧪 Le script `p9_pipeline.py` — quand l'utiliser ?

Une fois ton pipeline validé dans le notebook, lance la version production en batch :

```bash
aws emr add-steps \
    --cluster-id $(cat .last_cluster_id) \
    --region eu-west-3 \
    --steps "Type=Spark,Name=P9-Final-Run,ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,s3://mon-bucket-p9-tonprenom/code/p9_pipeline.py,s3://mon-bucket-p9-tonprenom/data/fruits-360/Test,s3://mon-bucket-p9-tonprenom/features-final,s3://mon-bucket-p9-tonprenom/pca-final]"
```

**Pourquoi le faire** : ton rapport doit montrer que tu sais industrialiser un traitement, pas seulement bricoler dans un notebook. C'est le critère **CE2/CE3 (script Spark exécuté sur machines cloud, écrivant directement dans S3)**.

---

## 📋 Mapping critères d'évaluation → preuves

| Critère | Comment c'est validé |
|---|---|
| Identifier briques d'archi Big Data | S3 + EMR/Spark + IAM + VPC — documenté dans le rapport |
| Outils cloud RGPD-compliant | Région `eu-west-3`, SSE-S3, IAM roles — cellule 10 du notebook |
| Fichiers entrée + sortie sur stockage cloud | `data/`, `features/`, `pca/` tous sur S3 |
| Scripts exécutés sur machines cloud | Cluster EMR (4 instances EC2) — capture console EMR |
| Script écrit sorties directement dans cloud | `result.write.parquet(PATH_PCA)` où `PATH_PCA = s3://...` |
| Traitements critiques au passage à l'échelle | I/O `binaryFile`, broadcast des poids, `pandas_udf` (vs `collect()`), partitionnement, mémoire executors |
| RGPD UE | Région UE pour S3 + EMR — capture montrant `eu-west-3` |
| Scripts Spark | `p9_pipeline.py` + notebook utilisent PySpark, Spark ML PCA, pandas_udf distribuées |
| Toute la chaîne cloud | S3 → EMR → S3, sans aucune étape locale |

---

## 💸 Estimation coût

Configuration par défaut (1 m5.xlarge + 3 m5.2xlarge en eu-west-3) : **~2 €/h**.

- Exploration notebook (2-3 h) : 4-6 €
- Run final batch (~30 min) : 1 €
- Stockage S3 (~5 Go) : <0.20 €/mois

**Total projet : 5-10 €** si tu es discipliné·e.

⚠️ Le piège : laisser le cluster up tout le week-end = ~100 €.

---

## 🐛 Debug rapide

| Symptôme | Solution |
|---|---|
| `bootstrap` failed | Logs dans `s3://.../emr-logs/<cluster-id>/node/<i-xxx>/bootstrap-actions/` |
| Notebook : kernel ne démarre pas | Cluster doit être en `WAITING`, `Livy` doit être dans `--applications` |
| `ImportError: tensorflow` worker | Bootstrap a échoué. Recréer le cluster. |
| Job lent | Spark UI → Stages → repérer sous-parallélisme ou shuffle énorme |
| `Access Denied` S3 | `EMR_EC2_DefaultRole` n'a pas accès au bucket. Check policy IAM. |
| Cluster oublié allumé | `aws emr list-clusters --active` régulièrement |
