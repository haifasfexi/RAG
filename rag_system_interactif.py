"""
Système RAG Interactif
- Charge des documents depuis des fichiers PDF ou JSON
- Permet de poser des questions interactivement
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import glob

# Pour lire les PDFs
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = True
    except ImportError:
        PDF_AVAILABLE = False
        print("[AVERTISSEMENT] Aucune bibliothèque PDF trouvée. Installez PyPDF2 ou pdfplumber:")
        print("  pip install PyPDF2")
        print("  ou")
        print("  pip install pdfplumber")

# ============================================================================
# STRUCTURE DE DONNÉES
# ============================================================================

@dataclass
class Document:
    """Représente un document dans la base de connaissances"""
    id: str
    content: str
    metadata: dict = None


# ============================================================================
# CHARGEUR DE DOCUMENTS
# ============================================================================

class DocumentLoader:
    """Charge des documents depuis différents formats"""
    
    @staticmethod
    def load_from_pdf(pdf_path: str) -> List[str]:
        """
        Charge le texte depuis un fichier PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Liste de chaînes de texte (une par page)
        """
        if not PDF_AVAILABLE:
            raise ImportError("Bibliothèque PDF non disponible. Installez PyPDF2 ou pdfplumber.")
        
        texts = []
        
        try:
            if 'USE_PDFPLUMBER' in globals() and USE_PDFPLUMBER:
                # Utilise pdfplumber
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            texts.append(text)
            else:
                # Utilise PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            texts.append(text)
        except Exception as e:
            print(f"[ERREUR] Impossible de lire le PDF {pdf_path}: {e}")
            return []
        
        return texts
    
    @staticmethod
    def load_from_json(json_path: str) -> List[dict]:
        """
        Charge des documents depuis un fichier JSON.
        
        Format attendu:
        [
            {"id": "doc1", "content": "texte...", "metadata": {...}},
            {"id": "doc2", "content": "texte...", "metadata": {...}}
        ]
        
        Ou format simple:
        [
            {"content": "texte..."},
            {"content": "texte..."}
        ]
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                print(f"[ERREUR] Format JSON invalide dans {json_path}")
                return []
        except Exception as e:
            print(f"[ERREUR] Impossible de lire le JSON {json_path}: {e}")
            return []
    
    @staticmethod
    def load_from_txt(txt_path: str) -> List[str]:
        """Charge le texte depuis un fichier .txt"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [content]
        except Exception as e:
            print(f"[ERREUR] Impossible de lire le fichier texte {txt_path}: {e}")
            return []
    
    @staticmethod
    def load_from_directory(directory: str) -> List[Document]:
        """
        Charge tous les documents depuis un dossier.
        
        Formats supportés:
        - *.pdf
        - *.json
        - *.txt
        
        Returns:
            Liste de documents
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            print(f"[ERREUR] Le dossier {directory} n'existe pas")
            return []
        
        documents = []
        
        # Charger les PDFs
        pdf_files = list(directory_path.glob("*.pdf"))
        for pdf_file in pdf_files:
            print(f"[CHARGEMENT] Lecture du PDF: {pdf_file.name}")
            texts = DocumentLoader.load_from_pdf(str(pdf_file))
            for i, text in enumerate(texts):
                doc = Document(
                    id=f"{pdf_file.stem}_page_{i+1}",
                    content=text,
                    metadata={"source": str(pdf_file), "page": i+1, "type": "pdf"}
                )
                documents.append(doc)
        
        # Charger les JSONs
        json_files = list(directory_path.glob("*.json"))
        for json_file in json_files:
            print(f"[CHARGEMENT] Lecture du JSON: {json_file.name}")
            data_list = DocumentLoader.load_from_json(str(json_file))
            for i, data in enumerate(data_list):
                if isinstance(data, dict):
                    doc = Document(
                        id=data.get("id", f"{json_file.stem}_{i+1}"),
                        content=data.get("content", ""),
                        metadata=data.get("metadata", {"source": str(json_file), "type": "json"})
                    )
                    documents.append(doc)
        
        # Charger les TXTs
        txt_files = list(directory_path.glob("*.txt"))
        for txt_file in txt_files:
            print(f"[CHARGEMENT] Lecture du fichier texte: {txt_file.name}")
            texts = DocumentLoader.load_from_txt(str(txt_file))
            for text in texts:
                doc = Document(
                    id=txt_file.stem,
                    content=text,
                    metadata={"source": str(txt_file), "type": "txt"}
                )
                documents.append(doc)
        
        print(f"[OK] {len(documents)} documents chargés depuis {directory}")
        return documents


# ============================================================================
# BASE DE DONNÉES VECTORIELLE
# ============================================================================

class SimpleVectorStore:
    """Base de données vectorielle avec sauvegarde"""
    
    def __init__(self, storage_dir: str = "rag_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.documents_file = self.storage_dir / "documents.json"
        self.embeddings_file = self.storage_dir / "embeddings.npy"
        
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.load_from_disk()
    
    def add_document(self, doc: Document, embedding: List[float]):
        """Ajoute un document et sauvegarde"""
        self.documents.append(doc)
        self.embeddings.append(embedding)
        self.save_to_disk()
    
    def add_documents_batch(self, docs: List[Document], embeddings: List[List[float]]):
        """Ajoute plusieurs documents en une fois (plus rapide)"""
        self.documents.extend(docs)
        self.embeddings.extend(embeddings)
        self.save_to_disk()
    
    def save_to_disk(self):
        """Sauvegarde sur le disque"""
        documents_data = []
        for doc in self.documents:
            doc_dict = {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            }
            documents_data.append(doc_dict)
        
        with open(self.documents_file, 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        if self.embeddings:
            embeddings_array = np.array(self.embeddings)
            np.save(self.embeddings_file, embeddings_array)
    
    def load_from_disk(self):
        """Charge depuis le disque"""
        if self.documents_file.exists():
            with open(self.documents_file, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            self.documents = []
            for doc_dict in documents_data:
                doc = Document(
                    id=doc_dict["id"],
                    content=doc_dict["content"],
                    metadata=doc_dict.get("metadata")
                )
                self.documents.append(doc)
            
            if len(self.documents) > 0:
                print(f"[CHARGEMENT] {len(self.documents)} documents chargés depuis le disque")
        
        if self.embeddings_file.exists():
            embeddings_array = np.load(self.embeddings_file)
            self.embeddings = embeddings_array.tolist()
            if len(self.embeddings) > 0:
                print(f"[CHARGEMENT] {len(self.embeddings)} embeddings chargés depuis le disque")
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Document, float]]:
        """Recherche les documents les plus similaires"""
        similarities = []
        
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcule la similarité cosinus"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def clear_all(self):
        """Supprime tous les documents"""
        self.documents = []
        self.embeddings = []
        if self.documents_file.exists():
            self.documents_file.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()
        print("[SUPPRESSION] Tous les documents ont été supprimés.")


# ============================================================================
# GÉNÉRATEUR D'EMBEDDINGS
# ============================================================================

class SimpleEmbedder:
    """Générateur d'embeddings simplifié"""
    
    def embed(self, text: str) -> List[float]:
        """Génère un embedding pour un texte"""
        text_lower = text.lower()
        embedding = [0.0] * 128
        
        for i, char in enumerate(text_lower[:128]):
            if char.isalnum():
                embedding[i % 128] += ord(char) / 1000.0
        
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour plusieurs textes"""
        return [self.embed(text) for text in texts]


# ============================================================================
# MODÈLE DE LANGAGE
# ============================================================================

class SimpleLLM:
    """Simulation d'un modèle de langage"""
    
    def generate(self, context: str, query: str) -> str:
        """Génère une réponse basée sur le contexte et la requête"""
        context_lower = context.lower()
        query_lower = query.lower()
        
        # Réponse basée sur le contexte récupéré
        # En production, on utiliserait GPT-4, Claude, etc.
        return (f"Basé sur les documents récupérés, voici une réponse à votre question "
               f"'{query}':\n\n"
               f"Contexte récupéré:\n{context[:500]}...\n\n"
               f"[NOTE: Ceci est une simulation. En production, un vrai LLM générerait "
               f"une réponse plus élaborée basée sur ce contexte.]")


# ============================================================================
# SYSTÈME RAG COMPLET
# ============================================================================

class RAGSystem:
    """Système RAG interactif"""
    
    def __init__(self, storage_dir: str = "rag_storage"):
        self.vector_store = SimpleVectorStore(storage_dir=storage_dir)
        self.embedder = SimpleEmbedder()
        self.llm = SimpleLLM()
    
    def load_documents_from_directory(self, directory: str, reindex: bool = False):
        """
        Charge des documents depuis un dossier et les indexe.
        
        Args:
            directory: Chemin vers le dossier contenant les fichiers
            reindex: Si True, supprime les anciens documents avant d'indexer
        """
        if reindex:
            self.vector_store.clear_all()
        
        # Charger les documents depuis le dossier
        documents = DocumentLoader.load_from_directory(directory)
        
        if not documents:
            print("[ERREUR] Aucun document trouvé dans le dossier")
            return
        
        # Générer les embeddings
        print("[INDEXATION] Génération des embeddings...")
        contents = [doc.content for doc in documents]
        embeddings = self.embedder.embed_batch(contents)
        
        # Ajouter à la base (en batch pour plus de rapidité)
        print("[INDEXATION] Ajout des documents à la base...")
        self.vector_store.add_documents_batch(documents, embeddings)
        
        print(f"[OK] {len(documents)} documents indexés avec succès!\n")
    
    def add_documents(self, documents: List[Document]):
        """Ajoute des documents manuellement"""
        contents = [doc.content for doc in documents]
        embeddings = self.embedder.embed_batch(contents)
        self.vector_store.add_documents_batch(documents, embeddings)
        print(f"[OK] {len(documents)} documents ajoutés!\n")
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Traite une question et génère une réponse"""
        if len(self.vector_store.documents) == 0:
            return "[ERREUR] Aucun document indexé. Chargez d'abord des documents."
        
        print(f"[QUESTION] {question}\n")
        
        # Encoder la requête
        query_embedding = self.embedder.embed(question)
        
        # Rechercher des documents pertinents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        print(f"[DOCUMENTS] {len(results)} documents pertinents trouvés:")
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "inconnu") if doc.metadata else "inconnu"
            print(f"   {i}. [Score: {score:.3f}] {Path(source).name if source != 'inconnu' else source}")
        print()
        
        # Construire le contexte
        context = "\n\n".join([doc.content for doc, _ in results])
        
        # Générer la réponse
        answer = self.llm.generate(context, question)
        
        return answer
    
    def interactive_mode(self):
        """Mode interactif pour poser des questions"""
        print("=" * 70)
        print("SYSTÈME RAG - MODE INTERACTIF")
        print("=" * 70)
        print()
        print(f"Documents indexés: {len(self.vector_store.documents)}")
        print()
        print("Commandes:")
        print("  - Tapez votre question et appuyez sur Entrée")
        print("  - Tapez 'quit' ou 'exit' pour quitter")
        print("  - Tapez 'clear' pour supprimer tous les documents")
        print("  - Tapez 'stats' pour voir les statistiques")
        print()
        print("-" * 70)
        print()
        
        while True:
            try:
                question = input("Votre question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nAu revoir!")
                    break
                
                if question.lower() == 'clear':
                    confirm = input("Êtes-vous sûr? (oui/non): ").strip().lower()
                    if confirm == 'oui':
                        self.vector_store.clear_all()
                        print("[OK] Documents supprimés\n")
                    continue
                
                if question.lower() == 'stats':
                    print(f"\n[STATISTIQUES]")
                    print(f"  Documents indexés: {len(self.vector_store.documents)}")
                    print(f"  Embeddings stockés: {len(self.vector_store.embeddings)}")
                    print(f"  Dossier de stockage: {self.vector_store.storage_dir}")
                    print()
                    continue
                
                # Traiter la question
                print()
                answer = self.query(question)
                print(answer)
                print("\n" + "-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nAu revoir!")
                break
            except Exception as e:
                print(f"[ERREUR] {e}\n")


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale"""
    print("=" * 70)
    print("SYSTÈME RAG INTERACTIF")
    print("=" * 70)
    print()
    
    # Créer le système RAG
    rag = RAGSystem(storage_dir="rag_storage")
    
    # Demander si on veut charger des documents
    print("Voulez-vous charger des documents depuis un dossier? (oui/non)")
    response = input("> ").strip().lower()
    
    if response in ['oui', 'o', 'yes', 'y']:
        directory = input("Chemin vers le dossier contenant les fichiers (PDF/JSON/TXT): ").strip()
        if directory:
            reindex = input("Réindexer tous les documents? (oui/non) [non]: ").strip().lower() == 'oui'
            rag.load_documents_from_directory(directory, reindex=reindex)
        else:
            print("[INFO] Aucun dossier spécifié. Utilisation des documents existants.")
    
    # Démarrer le mode interactif
    rag.interactive_mode()


if __name__ == "__main__":
    main()

