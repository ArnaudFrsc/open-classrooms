# Note de veille : Classification de texte à l'ère des LLM

## 1. Démarche et périmètre de veille

L'objectif initial est une tâche **classique** : classification multi-classes d'articles de presse (HuffPost, 30 catégories) sur la base de texte court (headline + short_description). Le notebook de référence (Logistic Regression + TF-IDF) représente l'état de l'art « pré-2020 » pour ce type de tâche.

Depuis 2020, le paysage NLP a été profondément remodelé par les **grands modèles de langage (LLM)** et le cadre de pensée dit d'**in-context learning (ICL)**. La veille porte donc sur trois axes :

1. **L'In-Context Learning** : peut-on remplacer un classifieur entraîné par un LLM auquel on fournit la tâche en langage naturel ?
2. **Le retrieval-augmented prompting** : peut-on améliorer l'ICL en sélectionnant dynamiquement les exemples du prompt via une recherche de similarité ?
3. **Le déploiement local** : les LLM open-weights (Llama, Mistral, Qwen) servis par Ollama permettent-ils de rester souverain sur les données ?  

## 2. Sources consultées

### Source 1 : Brown et al. (2020), *Language Models are Few-Shot Learners*

**Points clés** :
- Introduit le cadre de pensée **few-shot learning par prompting** : un LLM suffisamment grand (GPT-3, 175 B paramètres) peut apprendre une tâche nouvelle à partir de quelques exemples (« démonstrations ») fournis dans le prompt, **sans aucune mise à jour de gradient**.
- Définit trois régimes :
  - **Zero-shot** : seule l'instruction est donnée, aucun exemple.
  - **One-shot** : un seul exemple.
  - **Few-shot** : *k* exemples (typiquement 10 à 100).
- Démontre empiriquement que la performance suit une **loi d'échelle (scaling law)** : pour de nombreuses tâches, l'accuracy en few-shot croît log-linéairement avec le nombre de paramètres du modèle.

**Détail mathématique** : on note `M(·)` la fonction de complétion du LLM (déterministe à température nulle), `D = {(x_1, y_1), …, (x_k, y_k)}` les *k* démonstrations, et `x*` l'entrée à classifier. La prédiction few-shot s'écrit :

$$\hat{y} = M\big(\,\text{prompt}(D, x^*)\,\big)$$

où `prompt(·)` est une fonction de mise en forme (template) concaténant l'instruction, les démonstrations et la requête. **Aucun argmin sur une fonction de perte n'est calculé** : c'est la différence fondamentale avec un classifieur entraîné. Toute la « connaissance » provient des poids pré-entraînés du LLM et du contexte fourni.

**Dans ce notebook** : Les variantes `zero_shot` et `few_shot` lors de l'évaluation de la classification par LLM

---

### Source 2 : Liu et al. (2022), *What Makes Good In-Context Examples for GPT-3?*

**Points clés** :
- Les auteurs montrent que la performance ICL **dépend très fortement du choix des démonstrations**. Avec des exemples mal choisis, GPT-3 peut être pire que le random ; avec des exemples bien choisis, il rivalise avec des modèles fine-tunés.
- Ils proposent **KATE** (Knn-Augmented in-conText Example selection) : pour chaque requête `x*`, on retrouve les *k* exemples du train set les plus proches dans un espace d'embedding, et on les utilise comme démonstrations.
- Gains rapportés : **+5 à +15 points d'accuracy** selon les tâches, vs. un échantillonnage aléatoire d'exemples.

**Dans ce notebook** : La variante `dynamic_few_shot` (TF-IDF + `get_dynamic_examples`), l'apport principal du PoC.

---

### Source 3 : Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*

**Points clés** :
- Introduit **RAG** : combiner une mémoire **paramétrique** (les poids du LLM) avec une mémoire **non paramétrique** (un index vectoriel sur un corpus externe).
- L'architecture originale couple BART avec un index FAISS sur 21 M de passages Wikipedia. Pour chaque question, le retriever sélectionne les top-*k* passages, qui sont injectés dans le contexte du générateur.
- Atteint l'état de l'art sur Natural Questions, TriviaQA et FEVER. Plus important : ouvre la voie à des systèmes dont la **connaissance est éditable sans ré-entraînement** (suffit de mettre à jour l'index).

**Dans ce notebook** : conceptuellement, la variante `dynamic_few_shot` est une RA-ICL où le « corpus » est le train set étiqueté.

---

### Source 4 : Dong et al. (2024), *A Survey on In-context Learning*

**Points clés** : Synthèse récente qui formalise l'In-Context Learning en trois sous-problèmes :
1. **Demonstration design** : combien d'exemples, comment les ordonner, comment les formater ?
2. **Demonstration selection** : tirage aléatoire, k-NN, retriever appris (Rubin et al., 2022).
3. **Score function** : argmax direct, calibration, distribution de probabilités sur les labels.

Le papier confirme un résultat important : l'**ordre des exemples** dans le prompt peut faire varier l'accuracy de plusieurs points. Les auteurs constatent qu'aucune stratégie ne domine en toutes circonstances, mais que **retrieval > random** est quasi-systématique.

**Dans ce notebook** : guide les choix pratiques (5 voisins, ordre par similarité décroissante).

---

### Source 5 : Llama Team @ Meta AI (2024), *The Llama 3 Herd of Models*

**Points clés** :
- Décrit l'entraînement de Llama 3 (jusqu'à 405 B paramètres) sur ~15 T tokens, avec une **fenêtre de contexte de 128 K tokens**.
- Le modèle **Llama 3.1 8B** (celui utilisé dans ce notebook) est la version compacte de la famille, conçue pour tourner sur du matériel modeste (4–5 Go en quantization 4-bit).
- Benchmarks rapportés : sur des tâches de classification et NLI, Llama 3.1 8B atteint des performances proches de GPT-3.5 (avec un facteur ~20× moins de paramètres).

**Dans ce notebook** : choix du modèle `llama3.1:8b` via Ollama. C'est le **compromis taille/qualité** retenu pour le PoC.

---

## 3. Synthèse de la veille : ce qui change par rapport au baseline

| Dimension | Approche classique (réf.) | Approche LLM (PoC) |
|---|---|---|
| **cadre de pensée** | Apprentissage supervisé | In-Context Learning |
| **Représentation** | TF-IDF (vecteurs creux) | Embeddings appris (denses, 4096 dim chez Llama 3) |
| **Phase d'entraînement** | Obligatoire (`fit`) | Aucune (zero/few-shot) |
| **Adaptation à de nouvelles classes** | Impossible sans ré-entraînement | Immédiate (modifier le prompt) |
| **Coût d'inférence** | < 1 ms / article | ~1,5–2 s / article |
| **Interprétabilité formelle** | Coefficients lisibles par classe | Limitée |