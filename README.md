# ![PCnewsGPT](assets/PCnewsGPT%20Logo.png)

*Eine deutschsprachige, KI-basierte Wissensabfrage*

Wir verwenden die aktuelle KI wie z.B. chatGPT oft komplett falsch – nämlich zum Abfragen von Wissen, als ein Orakel. Ihre eigentliche Stärke ist aber das Verständnis unserer Sprache, und auch das sprachlich gut mit Text antworten zu können. 

Warum hat die aktuelle KI diese Schwäche, obwohl sie ja große Teile des öffentlichen Internets gelernt hat?

+ Ihr Wissen ist mit jenem Zeitpunkt "eingefroren", mit dem sie fertig angelernt war – ein aktuelles Wissen kennt sie nicht. Und sie darf eigentlich von sich aus nicht im Internet suchen oder andere Systeme um Hilfe Bitten (z.B. chatGPT mit Hilfe von Plug-Ins darf das)
+ Die Art wie sie programmiert wurde zwingt sie, auf Fragen immer mit etwas zu antworten – damit oft etwas zu erfinden bzw. mit etwas komplett Falschem zu "halluzinieren". Ausser, sie ist spezifisch darauf angelernt, in genau diesen spezifischen Fällen mit "ich kann das nicht" zu antworten.
+ Sie ist im Wissen sehr auf den angloamerikanischen Sprachraum fokussiert ("Bias").

Ein besserer, zuverlässiger Ansatz wäre, dieser sprachgewandten KI eine eigene, faktenbasierte Wissensbasis zur Seite zu stellen. Und bei Fragen über dieses Wissen, der KI die passenden Inhalte aus dieser Wissensbasis als Ausgangsmaterialen zu liefern. D.h. sie nur zum Analysieren, Verdichten des Wissens, und zum Formulieren der Antwort zu verwenden. Inklusive dem Hinweis in der Antwort, dass die KI nichts zur Frage Passendes in der Wissensbasis gefunden hat, anstatt unbemerkt zu halluzinieren. Dieser Ansatz wird als "Retrieval Augmented Generation (RAG)" bezeichnet, und ist eine sehr aktiver KI-Anwendungsbereich.

***Als Einführung und weiterführende Erklärung gibts auch meinen [PC-News Artikel](./assets/PCnewsGPT%20-%20für%20PC-News.docx) (als .docx).***

Das Projekt PCnewsGPT versucht diese Idee in einer speziell auf deutschsprachigen Inhalt abgestimmten Lösung umzusetzen. Und diese Lösung läuft vollständig auf einem lokalen Computer. Damit wird das Wissen ausschliesslich lokal und vertraulich bzw. sicher verarbeitet. D.h. diese Lösung vermeidet vollständig, vertrauliche oder urheberrechtsgeschützte Inhalte der eigenen Wissensbasis ins öffentliche Internet zu übertragen.

PCnewsGPT wurde als Open-Source Python-Programme unter Apache-Lizenz realisiert und ist hier öffentlich verfügbar bzw. jederzeit anpassbar. Der Test der Programme erfolgte mit den Inhalten der ClubComputer.at/DigitalSociety.at Clubzeitschrift PC-News (als unredigierte PDFs). Daher der Name "PCnewsGPT".

***PCnewsGPT ist nur ein laufend weiterentwickelter Prototyp, jegliche Anwendung erfolgt auf eigene Gefahr.***

## Technische Details

KI-Wissensabfrage von lokal gespeicherten PDF-Dateien mittels [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag), vollständig am lokalen Computer.

PCnewsGPT besteht im Wesentlichen aus zwei in Python implementierten Programmteilen, samt einer Konfigurationsdatei:

+ `import.py` - importieren der PC-News PDF Dateien in eine lokale Wissensdatenbank
  + Derzeit orchestriert über das `langchain` KI-Framework
  + Importiert die Dateien als einzelne Textseiten und dazugehörigen Metadaten
    + Derzeit derzeit nur für Dateityp .pdf implementiert
      + mittels der Bibliothek `PyMuPDF`
      + Räumt dabei den Rohtext für bessere Lesbarkeit auf. z.B. unterteilt nach Seiten, ignoriert Leerseiten/Bildseiten, entfernt Ligaturen,...
      + ***Optimierungspotenzial:*** Mit "Microsoft Print to PDF" erzeugte Dateien speichern im hinterlegten Text alle Ligaturen als das gleiche Sonderzeichen. D.h. dieser Text wird für beim Import für die häufigtstverwendeten Wörter in lesbaren Text umgewandelt. Die dafür beim Import verwendete Ersetzungstabelle muss ggf. noch erweitert werden.
    + ***Erweiterungspotenzial:*** ergänzen um Dateitypen .txt, .doc,...
  + Analysiert und zerteilt die importierten Texte in Wissensfragmente (`chunks`) mittels `SpaCy` als Text-Splitter (ein Parameter)
    + Mit max. Längenvorgabe je Wissensfragment (als Parameter)
    + Es wird ein speziell für deutsche Inhalstbedeutung optimierter Zusatzalgorithmus verwendet (`de_core_news_lg`, als Parameter)
    + Metadaten bleiben erhalten und werden zur besseren Identifikation um Wissensfragment (chunk) Nummern ergänzt
    + ***Optimierungspotenzial:*** Textseiten per verbesserter `SpaCy`-Pipeline besser auf konsitente Wissensfragmente zerlegen. Ev. mehrseitige Artikel erkennen und entsprechend aufbereiten.
  + Verwendet `HuggingFace/SentenceTransformers` zum Generieren der Bedeutungs- bzw. Embedding-Vektoren. 
    + Auch hier wird ein speziell für internationale Inhalstbedeutung optimiertes Modell (`intfloat/multilingual-e5-large`, als Parameter) verwendet.
  + Verwendet `chromaDB` als Wissensbasis Datenbank/Vektor-DB zum Speichern der Embedding-Vektoren und Wissensfragmente (Text + Metadaten), mit eingebetteter `duckDB` als relationaler DB
  + Zwei Betriebsarten - Initialimport der Datenbank und nachträgliches Hinzufügen von Inhalten (aus zwei parametriebaren Verzeichnissen)

+ `abfrage.py` - KI-Abfrage dieser Wissensdatenbank mittels lokalem KI Modell
  + Verwendet die vom Import generierten Wissensbasis in der `chromaDB` Vektordatenbank und die gleichen Embeddingalgorithmen
  + Die Benutzerfrage wird in einen Embedding-Vektor umgewandelt, und mit diesem Vektor wird in der DB nach zur Benutzerfrage passenden Context-Inhalten gesucht. Die Datenbank liefert die besten n (konfigurierbar via Parameter) passenden Wissensfragmente, inkl. einer Distanzangabe. Zu weit wegliegende Ergebnisse werden ignoriert (Parametrierbar).
  + Neu: Context-Inhalten, welche der Frage am nächsten und untereinander näher als ein entsprechender Parameter (`MAX_RESORT_DISTANCE`) sind, werden nach Erzeugungsdatum umsortiert, und in der KI-Abfrage entsprechend priorisiert. Dies soll helfen zu vermeiden, daß veraltete Informationen verwendet werden.
  + Verwendet `llama.cpp` und ein lokales KI Sprachmodell
    + Das Generieren der Antwort erfolgt mit einem "prompt" bestehend aus den dazupassenden Contexttexten der Wissensdatenbank und der Benutzerfrage sowie generellen Anweisungen. Die generellen Anweisungen verhindern, daß die KI eigene Fakten generiert und sich an den Wissenskontext halten sollte.
    + Verwendet [OpenBuddy](https://openbuddy.ai) als speziell für Multilinguale Anwendungen bzw. [llama2-13b-german-assistant-V7](https://huggingface.co/flozi00/Llama-2-13b-german-assistant-v7) als speziell für deutschsprachige Anwendungen entwickeltes KI-Modell (Parameter)

+ `.env` - Konfigurationsdatei
  + Alle Parameter können alternativ auch als Environment-Variables übergeben werden
  + ***Die Parameter MODEL_THREADS (Anzahl CPUs), MODEL_GPU (1 für Apple-GPU oder Anzahl verwendeter GPU-Kerne) und MODEL_PROMPT_PER_S (Geschwindigkeitsschätzung) müssen ggf. an den verwendeten Rechner angepasst werden.***
  + Die Parameter HIDE_SOURCE und HIDE_SOURCE_DETAILS sind fürs Testen, welches Wissen verwendet wurde


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
  + Grafikseiten oder Seiten mit zu wenig Text (weniger als 80 Zeichen) werden ignoriert
  + Mit dem Programm `dumpDB` in `./tests/` kann der Inhalt der Wissensdatenbank zu Testzwecken ausgelesen werden (Tipp: ggf. redirect der Ausgabe in Analysedatei)
  + Bei Problemen ggf. bitte das komplette Wissensdatenbankverzeichnis (definiert in `PERSIST_DIRECTORY` in `.env`) löschen und neu importieren

+ `abfrage.py` fürs ***Wissensbasis Abfragen***

## Offenes/Verbesserungspotenzial

Das Hauptproblem der verwendeten Methode (RAG) ist Garbage-in-Garbage-out, d.h. dass die Antwortqualität sehr von der Qualität der Wissensdatenbank und von der Qualität bei der Auswahl der in der LLM-Abfrage verwendeten Textfragmente abhängt.

<ins>Generelle Ideen dazu:</ins>

+ Aktuell: der verwendete SpaCy Textsplitter muss optimiert werden!!
+ Neue Idee für verbessertes Textsuchen bei Abfragen: [Parent Document Retriever](https://www.youtube.com/watch?v=wQEl0GGxPcM) - das würde aber ein "cleaning" der Quelldokumente nach Artikeln bedingen.

+ Eine sehr gute Zusammenfassung von Ideen ist [10 Ways to Improve the Performance of Retrieval Augmented Generation Systems](https://medium.com/towards-data-science/10-ways-to-improve-the-performance-of-retrieval-augmented-generation-systems-5fa2cee7cd5c)
+ Garbage-in-garbage-out, d.h. die Datenqualität der Wissensbasis ist DAS Hauptproblem. Ideen dazu
  + Ev. Dokumentenanalyse mit LLM vor dem import - siehe Experimente in `./tests`
  + Ev. Optimierung der chunk-längen, overlaps und max_content_chunks - keine Programmänderung, sondern nur Parameteränderungen.
+ Mehr in meiner [`ImprovementConcepts.md`](./ImprovementConcepts.md) Ideensammlung
+ Der "prompt" in `abfrage.py` soll Halluzinationen vermeiden, Neues/Gutes priorisieren und kurz sein - es gibt dafür leider keine dt. Vorlagen/Ideen. Hier ist sicher einiges an Optimierungspotenzial.
+ Die aktuelle Implementierung ist auf Funktion/Qualität/Änderbarkeit optimiert und nicht auf Geschwindigkeit, das bezieht sich auch auf das verwendete Embeddingmodell.
  + `langchain` in `import.py` ist eigentlich unnötiger overhead, gehört ev. wegoptimiert (so wie in `abfrage.py`) 
