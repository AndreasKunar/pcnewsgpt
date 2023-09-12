# ![PCnewsGPT](assets/PCnewsGPT%20Logo.png)

*Eine deutschsprachige, KI-basierte Wissensabfrage*

Wir verwenden die aktuelle KI wie z.B. chatGPT oft komplett falsch – nämlich als ein Orakel. Ihre eigentliche Stärke ist aber das Verständnis unserer Sprache, und auch das sprachlich gut mit Text antworten zu können. Obwohl die KI große Teile des öffentlichen Internets gelernt hat, ist ihr spezifisches Wissen sehr beschränkt. Sie ist im Wissen nämlich sehr auf den angloamerikanischen Sprachraum fokussiert ("Bias"). Und ihr Wissen ist mit jenem Zeitpunkt eingefroren, mit dem sie fertig angelernt war – ein aktuelles Wissen kennt sie nicht. Aber noch schlimmer, die Art wie sie programmiert wurde zwingt sie, auf Fragen immer mit etwas zu antworten – damit oft etwas zu erfinden bzw. mit etwas komplett Falschem zu "halluzinieren".

Ein besserer, zuverlässiger Ansatz wäre, dieser sprachgewandten KI eine faktenbasierte Wissensbasis zur Seite zu stellen. Und bei Fragen über dieses Wissen, der KI die passenden Inhalte dieser Wissensbasis als Ausgangsmaterialen zu liefern. D.h. sie nur zum Analysieren, Verdichten des Wissens, und zum Formulieren der Antwort zu verwenden. Inklusive dem Hinweis in der Antwort, dass die KI nichts zur Frage Passendes in der Wissensbasis gefunden hat, anstatt unbemerkt zu halluzinieren.

Das Projekt PCnewsGPT versucht diese Idee in einer speziell auf deutschsprachigen Inhalt abgestimmten Lösung umzusetzen. Und diese Lösung vollständig auf einem lokalen Computer laufen zu lassen. Damit wird das Wissen komplett lokal und vertraulich bzw. sicher verarbeitet, ohne dem öffentlichen Internet neues KI-Lernmaterial mit ggf. vertraulichen oder urheberrechtsgeschützten Inhalten zu liefern.

PCnewsGPT wurde als Open-Source Python-Programme unter Apache-Lizenz realisiert und ist hier öffentlich verfügbar bzw. jederzeit anpassbar. Der Test erfolgte mit den Inhalten der ClubComputer.at/DigitalSociety.at Clubzeitschrift PC-News (als unredigierte PDFs). Daher der Name "PCnewsGPT".

***PCnewsGPT ist nur ein laufend weiterentwickelter Prototyp, jegliche Anwendung erfolgt auf eigene Gefahr.***

## Technische Details

KI-Wissensabfrage von lokal gespeicherten PDF-Dateien mittels [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag), ohne Cloud am lokalen Computer.

PCnewsGPT besteht im Wesentlichen aus zwei Python Programmen und einer Konfigurationsdatei:

+ `import.py` - importieren der PC-News PDF Dateien in eine lokale Wissensdatenbank
  + Python Code, orchestriert über das `langchain` KI-Framework
  + Importiert die Dateien mittels `langchain.document_loaders` (nur für Typ .PDF) in einen `langchain.docstore`
  + Räumt PDF Dateien auf (nach Seiten, entfernt Ligaturen,...)
  + Zerstückeln die importierten Dateien in Wissensfragmente mittels `langchain.text_splitter` und Typ `SpaCy`. Dabei wird ein speziell für deutsche Inhalstbedeutung optimierter Zusatzalgorithmus verwendet (`de_core_news_lg`).
  + Verwendet `langchain.embeddings` und `HuggingFace/SentenceTransformers` zum Generieren der Bedeutungs- bzw. Embedding-Vektoren. Auch hier mit einem speziell für internationale Inhalstbedeutung optimiertem Modell (`paraphrase-multilingual-mpnet-base-v`).
  + Verwendet `langchain.vectorstores` und `chromaDB` als Wissensdatenbank/Vektor-DB

+ `abfrage.py` - KI-Abfrage dieser Wissensdatenbank mittels lokalem LLaMa-basiertem KI Modell
  + Verwendet die vom Import generierten Wissensbasis in der `chroma` Vektordatenbank
  + Verwendet die gleichen `HuggingFace/SentenceTransformers` zum Generieren der Bedeutungs- bzw. Embedding-Vektoren für die Benutzerfrage wie der Import. Mit diesen Vektoren wird in der DB nach dazu passenden Context-Inhalten gesucht
  + Verwendet `llama.cpp` und ein auf llama basiertes KI Modell als KI Sprachmodell. Mit einem "promp" bestehend aus den Contexttexten und der Frage sowie generellen Anweisungen.
  + [OpenBuddy](https://openbuddy.ai) sind speziell für Multilinguale Anwendungen entwickelte KI-Modelle

+ `.env` - Konfigurationsdatei
  + Alle Parameter können auch als Environment-Variables übergeben werden bzw. haben gute Defaults

## Installation

Diese Installation wurde unter Apple M1/M2 basiertem macOS (in host-OS und VM) und Apple M1/M2 basierten Linux VMs (ubuntu) mit mindestens 16GB RAM getestet. Ältere Macs/PCs ohne moderne CPUs, mit weniger als 16GB RAM und ev. keinem aktuellen Grafikbeschleuniger sind für dies nicht geeignet! Als Hardwarekompatibilitätstest eignet sich z.B. die Installation und das Ausführen von [llama.cpp](https://github.com/ggerganov/llama.cpp) mit dem KI-Modell von Schritt 1.

1. LLaMa basiertes KI-Modell, z.B. `openbuddy-llama2-13b-v11.1.Q4_K_M.gguf` (ein multilinguales 13B bzw. 13 Milliaden Parameter Modell) von [huggingface.co/TheBloke](https://huggingface.co/TheBloke) in Verzeichnis `./models` installieren. Ggf. `.env` Konfiguration anpassen, falls ein anderes Modell verwendet wird.

2. PDF-Dateien der Wissensdatenbank (z.B. [PC-news PDF-Ausgabe](http://d.pcnews.at/_pdf/n178.pdf)) in Verzeichnis `./source_documents` kopieren. Ggf. `.env` Konfiguration anpassen, falls ein anderes Verzeichnis verwendet wird.

3. Lösungen in Python (wie PCnewsGPT) verwenden viele Python Bibliotheken. Um die unterstützte Python-Version und Bibliotheksversionsinkompatibilitäten zu vermeiden, wird eine Python "environment" empfohlen, z.B. verwaltet über `conda`.

    + Eine `conda` Python Environment anlegen und **aktivieren**:

    ```shell
    conda create -n pcnewsgpt python=3.10.11 -y
    conda activate pcnewsgpt
    ```

    + PCnewsGPT benötigt aktuell eine 3.10er Version von Python, da `pytorch`, welches von den verwendeten `sentenceTransformers` benötigt wird, aktuell nur bis 3.10 kompatibel ist.

4. Nötige Python Bibliotheken für `abfrage.py` in der jeweiligen passenden Version installieren.

    + Aktuelle Bibliothek `llama-cpp-python` installieren

    ```shell
    pip install llama-cpp-python
    ```

    + ***Alternativ für Apple Silicon basierte Hardware in macOS für GPU-Verwendung*** mittels:

    ```shell
    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
    ```

    + ***Alternativ für Apple Silicon basierte Hardware in Linux VMs oder Docker*** (erfordert Install-Patch) mittels:

    ```shell
    CMAKE_ARGS="-DUNAME_M=arm64 -DUNAME_P=arm -DLLAMA_NO_METAL=1" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
    ```

    + Dann die weiteren Bibliotheksabhängigkeiten installieren (`chromaDB` installiert dabei viele eigene Abhängigkeiten, wie z.B. `pyTorch`, Huggingface `transformers`, ...):

    ```shell
    pip install pydantic==1.10.12
    pip install chromadb==0.3.23
    pip install pandas==1.5.3
    ```

    + Bei der ersten Ausführung laden die `SentenceTransformers` entsprechende Dateien/Modelle selbstätig aus dem Internet herunter. Ab dann ist die Lösung voll Offlinefähig.

5. Falls in dieser Installation auch `import.py` ausgeführt werden soll (die Abfrage alleine ist simpler und benötigt weniger Python-Bibliotheken), die nötige Bibliotheken dafür in der jeweiligen passenden Version installieren.

    ```shell
    pip install -r requirements.txt
    python -m spacy download de_core_news_lg
    ```

## Ausführen

***Vor jedem neuen Start bitte nicht vergessen, die conda Umgebung ggf. über `conda activate pcnewsgpt` zu aktivieren!***

+ alle Einstellungen sind in `.env`

+ `import.py` fürs ***Wissensbasis Importieren***
  + Wenn eine Wissensdatenbank vorhanden ist, werden die Dateien aus dem `APPEND_DIRECTORY` (definiert in `.env`) dazuimportiert und danach ins `SOURCE_DIRECTORY` (auch definiert in `.env`) verschoben - damit ist jederzeit eine Erweiterung des Wissens möglich!
  + Sollte keine Wissendatenbenk vorhanden sein, so wird eine leere erzeugt und die Dateien aus `SOURCE_DIRECTORY` importiert
  + Mit dem Programm `dumpDB` kann der Inhalt der Wissensdatenbank zu Testzwecken ausgelesen werden (Tipp: ggf. redirect der Ausgabe in Analysedatei)
  + Bei Problemen ggf. bitte das komplette Wissensdatenbankverzeichnis (definiert in `PERSIST_DIRECTORY` in `.env`) löschen und neu importieren

+ `abfrage.py` fürs ***Wissensbasis Abfragen***

## Offenes/Verbesserungspotenzial

Das Hauptproblem der verwendeten Methode (RAG) ist Garbage-in-Garbage-out, d.h. dass die Antwortqualität sehr von der Qualität der Wissensdatenbank und von der Qualität bei der Auswahl der in der LLM-Abfrage verwendeten Textfragmente abhängt.

<ins>Ideen/Papers dazu:</ins>

+ Markieren des Ursprungsdatums der Datenquelle. Damit neuere Daten bei der Abfrage höher gewichten, und nicht mehr aktuelle Daten eher ignorieren.
+ [Improving Retrieval-Augmented Large Language Models via Data Importance Learning](https://arxiv.org/pdf/2307.03027.pdf)
+ [Improving RAG Answer Quality with Re-ranker](https://medium.com/towards-generative-ai/improving-rag-retrieval-augmented-generation-answer-quality-with-re-ranker-55a19931325)
+ [Meine `ImprovementConcepts.md` Sammlung](./ImprovementConcepts.md)

### Verbesserungsideen Import

+ Datenqualitätsverbesserungen
  + Ev. zusätzliche Zeichenersetzungen für bessere Lesbarkeit finden
  + Ev. optimieren der chunk-längen, overlaps und max_content_chunks - keine Programmänderung, sondern nur Parameteränderungen.
  + Dokumenthierarchie Parsing wäre Ideal - Benötige aber allgemeingültigen Algorithmus
    + ein analysierbares Inhaltsverzeichnis und das finden der dazugehörigen Seite(n) wäre ideal - Algorithmus?
    + Die PC-news Titelseiten tragen kein Wissen bei, ggehören ev. ignoriert. Algorithmus diese generisch zu finden?
+ Getestet für .PDFs, andere Dateiformate gehören noch ggf. hinzugefügt

+ Die aktuelle Version ist auf Funktion/Qualität/Änderbarkeit optimiert und langsam.
+ `langchain` in `import.py` ist eigentlich unnötiger overhead, gehört ev. wegoptimiert (so wie in `abfrage.py`) sobald import Funktional komplett "stabilisiert" ist.

### Verbesserungsideen Abfrage

+ Der "prompt" soll Halluzinationen vermeiden und kurz sein - es gibt dafür leider keine dt. Vorlagen/Ideen
+ Das initiale Analysieren des mit jeder Frage neu generierten "prompt" and die KI durch llama.cpp dauert ziemlich lang und ist ohne Fortschrittsanzeige (deshalb das "Antwort - bitte um etwas Geduld"). Ev. gibts hier Methoden einer Fortschrittsanzeige.
