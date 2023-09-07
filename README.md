# PC-News Inhaltsabfragen mittels KI

KI-Wissensabfrage von lokal gespeicherten PC-News (als PDF-Dateien) mittels [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag) - vollkommen ohne Cloud am lokalen Computer, aber leicht änderbar auf mix- oder cloud-Betrieb durch das verwendete [langchain](https://github.com/langchain-ai/langchain) Framework.

Dies ist eine Technologiedemonstration basierend auf Ideen von [imartinez/privateGPT](https://github.com/imartinez/privateGPT). Sie besteht im Wesentlichen aus zwei Python Programmen und einer Konfigurationsdatei:

+ `import.py` - importieren der PC-News PDF Dateien in eine lokale Wissensdatenbank
  + Python Code, orchestriert über das `langchain` KI-Framework
  + importieren der Dateien mittels `langchain.document_loaders` (nur für Typ .PDF) in `langchain.docstore`
  + räumt PDF Dateien auf (nach Seiten, entfernt Ligaturen,...)
  + zerstückeln in Wissenshäppchen mittels `langchain.text_splitter` und Typ `SpaCy` (für deutsche Inhalstbedeutung)
  + `langchain.embeddings` und `HuggingFace/SentenceTransformers` zum Generieren der Bedeutungs- bzw. Embedding-Vektoren
  + `langchain.vectorstores` und `chromaDB` als Wissensdatenbank/Vektor-DB
+ `abfrage.py` - KI-Abfrage dieser Wissensdatenbank mittels lokalem LLaMa-basiertem KI Modell
  + Python Code, orchestriert über das `langchain` KI-Framework
  + mittels Wissensbasis in chroma vactorDB und llama.cpp basiertem KI Modell
+ `.env` - Konfigurationsdatei

## Installation (unter macOS)

Dieses Projekt verwendet `poetry` für die Verwaltung der Abhängigkeiten.

Über `conda` eine virtuelle Python Umgebung anlegen und aktivieren:

```shell
conda create -n pcnewsgpt python=3.10.9
conda activate pcnewsgpt
```

llama-cpp-python` manuell installieren - z.B. in Apple Silicon basierter VM mittels

```shell
CMAKE_ARGS="-DUNAME_M=arm64 -DUNAME_p=arm -DLLAMA_NO_METAL=1" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
```

Dann die Abhängigkeiten Installieren:

```shell
poetry --without-hashes export -f requirements.txt --output requirements.txt
pip install -r requirements.txt
pip install spacy
python -m spacy download de_core_news_lg
```

LLaMa basiertes KI-Modell, z.B. `openbuddy-llama2-13b-v11.1.Q4_K_M.gguf` (ein Multilinguales Modell) von [huggingface.co/TheBloke](https://huggingface.co/TheBloke) in `./models` installieren, ggf. `.env` Konfiguration anpassen.

&nbsp;

## Ausführen

Vor jedem neuen Start bitte nicht vergessen, ggf. die Python Umgebung über `conda activate pcnewsgpt` zu aktivieren!

### `import.py` - Wissensbasis Importieren

### TODO Beschreibung

### `abfrage.py` - Wissensbasis Abfragen

### more TODO Beschreibung
