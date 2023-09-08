# PCnewsGPT – eine deutschsprachige KI-basierte Wissensabfrage

Wir verwenden die aktuelle KI wie z.B. chatGPT oft komplett falsch – nämlich als ein Orakel. Ihre eigentliche Stärke ist aber das Verständnis unserer Sprache, und auch das sprachlich gut mit Text antworten zu können. Obwohl die KI große Teile des öffentlichen Internets gelernt hat, ist ihr spezifisches Wissen sehr beschränkt. Sie ist im Wissen sehr auf den angloamerikanischen Sprachraum fokussiert. Und das Wissen ist mit jenem Zeitpunkt eingefroren, zu dem sie fertig angelernt war – ein aktuelles Wissen kennt sie nicht. Und noch schlimmer, die Art wie sie programmiert wurde zwingt sie, auf Fragen immer mit etwas zu antworten – damit etwas zu erfinden bzw. mit etwas komplett Falschem zu halluzinieren.

Ein besserer, zuverlässiger Ansatz wäre, dieser sprachgewandten KI eine faktenbasierte Wissensbasis zur Seite zu stellen. Und bei Fragen über dieses Wissen, der KI die passenden Inhalte dieser Wissensbasis als Ausgangsmaterialen zu liefern, sie nur zum Analysieren, Verdichten des Wissens, und zum Formulieren der Antwort zu verwenden. Inklusive der Antwort, dass die KI nichts zur Frage Passendes in der Wissensbasis gefunden hat, anstatt hier zu halluzinieren.

Das Projekt PCnewsGPT versucht diese Idee in einer speziell auf deutschsprachigen Inhalt abgestimmten Lösung umzusetzen. Und diese Lösung rein auf einem lokalen Computer laufen zu lassen. Damit das Wissen komplett lokal und vertraulich bzw. sicher verarbeitet wird – d.h. ohne öffentlich neues, zukünftiges KI-Lernmaterial mit eventuell vertraulichen, urheberrechtsgeschützten Inhalten zu liefern. PCnewsGPT wurde als Open-Source Python-Programme unter Apache-Lizenz realisiert und ist hier öffentlich verfügbar bzw. jederzeit anpassbar.

***PCnewsGPT ist aber nur ein laufend weiterentwickelter Prototyp, jegliche Anwendung erfolgt auf eigene Gefahr.***

## Technische Details

KI-Wissensabfrage von lokal gespeicherten PDF-Dateien mittels [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag) - vollkommen ohne Cloud am lokalen Computer, aber leicht änderbar auf mix- oder cloud-Betrieb durch eine Implementierung in Python.

PCnewsGPT besteht im Wesentlichen aus zwei Python Programmen und einer Konfigurationsdatei:

+ `import.py` - importieren der PC-News PDF Dateien in eine lokale Wissensdatenbank
  + Python Code, orchestriert über das `langchain` KI-Framework
  + importieren der Dateien mittels `langchain.document_loaders` (nur für Typ .PDF) in einen `langchain.docstore`
  + räumt PDF Dateien auf (nach Seiten, entfernt Ligaturen,...)
  + zerstückeln in Wissenshäppchen mittels `langchain.text_splitter` und Typ `SpaCy` (für deutsche Inhalstbedeutung)
  + verwendet `langchain.embeddings` und `HuggingFace/SentenceTransformers` zum Generieren der Bedeutungs- bzw. Embedding-Vektoren
  + verwendet `langchain.vectorstores` und `chromaDB` als Wissensdatenbank/Vektor-DB
+ `abfrage.py` - KI-Abfrage dieser Wissensdatenbank mittels lokalem LLaMa-basiertem KI Modell
  + Python Code
  + mittels der von `import.py` generierten Wissensbasis in der `chroma` Vektordatenbank
  + verwendet `HuggingFace/SentenceTransformers` zum Generieren der Bedeutungs- bzw. Embedding-Vektoren
  + verwendet ein llama.cpp basiertes KI Modell
+ `.env` - Konfigurationsdatei

## Installation (getestet unter macOS und Linux mit modernem Rechner und 16GB RAM)

1. LLaMa basiertes KI-Modell, z.B. `openbuddy-llama2-13b-v11.1.Q4_K_M.gguf` (ein multilinguales 13B / 13 Mrd. Parameter Modell) von [huggingface.co/TheBloke](https://huggingface.co/TheBloke) in Verzeichnis/Ordner `./models` installieren. Ggf. `.env` Konfiguration anpassen, falls anderes Modell verwendet wird.

2. PDF-Dateien der Wissensdatenbank (z.B. [PC-news PDF-Ausgabe](http://d.pcnews.at/_pdf/n178.pdf))in Verzeichnis/Ordner `./source_documents` kopieren.

3. Nötige Python Bibliotheken installieren. Dieses Projekt verwendet momentan `poetry` für die Verwaltung der Abhängigkeiten. Dies funktioniert aber nicht so wie gedacht, und gehört überarbeitet.

    + Über `conda` eine virtuelle Python Umgebung anlegen und aktivieren:

    ```shell
    conda create -n pcnewsgpt python=3.10.9
    conda activate pcnewsgpt
    ```

    + llama-cpp-python` manuell installieren - z.B. in Apple Silicon basierter VM mittels

    ```shell
    CMAKE_ARGS="-DUNAME_M=arm64 -DUNAME_p=arm -DLLAMA_NO_METAL=1" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
    ```

    + Dann die Abhängigkeiten Installieren:

    ```shell
    poetry --without-hashes export -f requirements.txt --output requirements.txt
    pip install -r requirements.txt
    pip install spacy
    python -m spacy download de_core_news_lg
    ```

    + Bei der allerersten Ausführung laden die `SentenceTransformers` die entsprechenden Dateien/Modelle selbstätig aus dem Internet herunter.

## Ausführen

Vor jedem neuen Start bitte nicht vergessen, die conda Umgebung ggf. über `conda activate pcnewsgpt` zu aktivieren!

### `import.py` - Wissensbasis Importieren

Offenes/Verbesserungspotenzial:

+ Vor dem Start bitte das Datenbankverzeichnis `./db` löschen
+ PDF-Umwandlung ist langsam, gehört optimiert
+ PDF-Umwandlung hat noch Verbesserungspotenzial
+ `langchain` ist hier eigentlich unnötiger overhead

### `abfrage.py` - Wissensbasis Abfragen

Offenes/Verbesserungspotenzial:

+ Deutscher `PROMPT` muss noch optimiert werden
+ `langchain` ist hier eigentlich unnötiger overhead, ohne dieses wäre die Abfrage deutlich schneller
