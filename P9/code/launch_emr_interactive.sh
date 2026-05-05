#!/bin/bash
# ============================================================
# Création d'un cluster EMR pour usage NOTEBOOK INTERACTIF
# (EMR Studio, exploration, itération)
# ============================================================
# Différences avec launch_emr.sh standard :
#   - PAS de --steps : le cluster reste up, on lui envoie du code via Studio
#   - PAS de --auto-terminate : on garde le cluster jusqu'à arrêt manuel
#   - IdleTimeout=3600 : sécurité — le cluster s'éteint si inactif 1h
#
# 💰 Coût : ~2 $/h tant que le cluster est up.
# 🇪🇺 RGPD : région EU obligatoire (eu-west-3 = Paris).
# ============================================================

# ===== À ADAPTER =====
BUCKET="mon-bucket-p9-tonprenom"
REGION="eu-west-3"
KEY_NAME="ma-keypair"
SUBNET_ID="subnet-xxxxxxxx"
LOG_URI="s3://${BUCKET}/emr-logs/"

# ===== 1. Création du cluster =====
echo "→ Création du cluster EMR (mode interactif)..."

CLUSTER_ID=$(aws emr create-cluster \
    --name "P9-Fruits-Interactive" \
    --release-label emr-7.13.0 \
    --region ${REGION} \
    --applications Name=Spark Name=Hadoop Name=JupyterEnterpriseGateway Name=Livy \
    --ec2-attributes "KeyName=${KEY_NAME},SubnetId=${SUBNET_ID}" \
    --service-role EMR_DefaultRole \
    --instance-groups \
        "InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge" \
        "InstanceGroupType=CORE,InstanceCount=3,InstanceType=m5.2xlarge" \
    --bootstrap-actions \
        "Path=s3://${BUCKET}/code/bootstrap.sh,Name=Install-Python-Libs" \
    --log-uri "${LOG_URI}" \
    --auto-termination-policy IdleTimeout=3600 \
    --query 'ClusterId' \
    --output text)

echo ""
echo "✓ Cluster lancé : ${CLUSTER_ID}"
echo ""
echo "  Attendre l'état 'WAITING' (3-5 min de provisioning + bootstrap) :"
echo "  aws emr wait cluster-running --cluster-id ${CLUSTER_ID} --region ${REGION}"
echo ""
echo "  Console :"
echo "  https://${REGION}.console.aws.amazon.com/emr/home?region=${REGION}#/clusterDetails/${CLUSTER_ID}"
echo ""
echo "  Pour terminer manuellement :"
echo "  aws emr terminate-clusters --cluster-ids ${CLUSTER_ID} --region ${REGION}"

# Garder l'ID en local pour les commandes suivantes
echo "${CLUSTER_ID}" > .last_cluster_id
echo ""
echo "  (ID sauvegardé dans .last_cluster_id)"
