# Version 3 : Système RAG Interactif

## 📋 Description

Version complète avec chargement de fichiers PDF/JSON/TXT et mode interactif pour poser des questions.

## ✨ Caractéristiques

- ✅ Charge des fichiers PDF, JSON et TXT
- ✅ Mode interactif pour poser des questions
- ✅ Sauvegarde automatique des documents
- ✅ Support de plusieurs formats de fichiers

## 📁 Fichiers Inclus

- `rag_system_interactif.py` - Script principal
- `GUIDE_UTILISATION.md` - Guide complet d'utilisation
- `README_SYSTEME_INTERACTIF.md` - Guide rapide
- `documents/` - Dossier avec des exemples de fichiers

## 🚀 Installation

```bash
pip install numpy PyPDF2
```

**Note :** Vous pouvez aussi utiliser `pdfplumber` au lieu de `PyPDF2` :
```bash
pip install numpy pdfplumber
```

## 💻 Utilisation

### 1. Préparer vos documents

Placez vos fichiers dans le dossier `documents/` :
- Fichiers PDF : `*.pdf`
- Fichiers JSON : `*.json`
- Fichiers texte : `*.txt`

### 2. Lancer le système

```bash
python rag_system_interactif.py
```

### 3. Charger les documents

Quand demandé :
```
Voulez-vous charger des documents depuis un dossier? (oui/non)
> oui
Chemin vers le dossier: documents
Réindexer tous les documents? (oui/non) [non]: non
```

### 4. Poser des questions

```
Votre question: Qu'est-ce que le RAG ?
```

## 📋 Format JSON Accepté

```json
[
  {
    "id": "doc1",
    "content": "Le texte de votre document...",
    "metadata": {
      "auteur": "Nom",
      "date": "2024"
    }
  }
]
```

## 💬 Commandes Disponibles

- **Tapez votre question** : Pour poser une question
- **`quit`** ou **`exit`** : Quitter le programme
- **`clear`** : Supprimer tous les documents
- **`stats`** : Voir les statistiques

## 📂 Où sont Sauvegardés les Documents ?

Les documents sont sauvegardés dans :
- `rag_storage/documents.json` - Les documents (texte)
- `rag_storage/embeddings.npy` - Les vecteurs (embeddings)

## 📚 Documentation

- Consultez `GUIDE_UTILISATION.md` pour le guide complet
- Consultez `README_SYSTEME_INTERACTIF.md` pour le guide rapide

## 🔄 Pour Aller Plus Loin

Si vous voulez :
- **Version simple** → Utilisez `version1_simple/`
- **Version avec sauvegarde seulement** → Utilisez `version2_sauvegarde/`

