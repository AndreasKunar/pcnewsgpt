"""
*** PCnewsGPT Wissensimporter - import.py ***
    Änderung: V0.1.5.x - PyMuPDF direkt eingebunden
    Änderung: V0.1.4.x - PyMuPDFLoader ersetzt pdfMiner.six
    Änderung: V0.1.3.x - append als funktion, besser lesbare, verständliche chunk-texte
"""

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ
from ast import literal_eval
load_dotenv()
persist_directory = os_environ.get('PERSIST_DIRECTORY','db')
embeddings_model_name = os_environ.get('EMBEDDINGS_MODEL_NAME','paraphrase-multilingual-mpnet-base-v2')
source_directory = os_environ.get('SOURCE_DIRECTORY','source_documents')
append_directory = os_environ.get('APPEND_DIRECTORY','append_documents')
text_splitter_name = os_environ.get('TEXT_SPLITTER','RecursiveCharacterTextSplitter')
text_splitter_parameters = literal_eval(os_environ.get('TEXT_SPLITTER_PARAMETERS','{"chunk_size": 500, "chunk_overlap": 50}'))

"""
Initial banner Message
"""
print("\nPCnewsGPT Wissensimporter V0.1.5\n")

"""
PDF-Import via PyMuPDF with some more processing, 
    ignore any pages with full-page-sized images (covers, ads, etc.)
"""
from langchain.docstore.document import Document as Langchain_Document
import fitz as PyMuPDF
from regex import sub as regex_sub
def myPDFLoader (fname) -> Langchain_Document:
    return_docs = []                  # return value
    with PyMuPDF.open(fname) as doc:  # open document

        # simplified metadata for full document
        doc_metadata = {
            'source':       fname,
            'format':       doc.metadata.get('format'), 
            'author':       doc.metadata.get('author'), 
            'producer':     doc.metadata.get('producer'),  
            'creationDate': doc.metadata.get('creationDate'),  
        }

        # process all document pages (page.number+1 for "real" numbering) 
        for page in doc:

            # *** check if full-page image

            # calculate the page's width and height in cm
            page_w = round((page.mediabox.x1 - page.mediabox.x0)/72*2.54,2)     # 72 dpi, 2,54 cm / inch
            page_h = round((page.mediabox.y1 - page.mediabox.y0)/72*2.54,2)

            # get all images and calculate concatenated img width x height
            image_infos = page.get_image_info()
            i=None
            for image_info in image_infos:
                x = image_info.get('bbox')[0]/72*2.54                           # 72 dpi, 2,54 cm / inch
                y = image_info.get('bbox')[1]/72*2.54
                w = (image_info.get('bbox')[2]-image_info.get('bbox')[0])/72*2.54
                h = (image_info.get('bbox')[3]-image_info.get('bbox')[1])/72*2.54
                # 1st or next image
                if i is None:
                    imgs=[{'x':x, 'y':y, 'w': w,'h':h }]
                    i=0
                else:
                    prev_img = imgs[i]
                    # try concatenate images vertically (same x and w)
                    if (x == prev_img.get('x')) and (w == prev_img.get('w')) and (abs(y - prev_img.get('y') - prev_img.get('h')) < 0.1):
                        # extend image height
                        imgs[i]['h'] += h
                    # try concatenate horizontally (same y and h)              
                    elif (y == prev_img.get('y')) and (h == prev_img.get('h')) and (abs(x - prev_img.get('x') - prev_img.get('w')) < 0.1):
                        # extend image width
                        imgs[i]['w'] += w
                    else:
                        imgs.append({'x':x, 'y':y, 'w': w,'h':h })
                        i=i+1
            
            # check if a concetenated image spans 95% of page
            fullpage_img = False
            for img in imgs:
                if (img.get('w') > page_w*0.95) and (img.get('h') > page_h*0.95):   # 95% of page width and height is enough
                    fullpage_img = True
                    print(f"{fname} Seite:{page.number+1} ignoriert, da Ganzseitenbild {round(img['w'],2)}x{round(img['h'],2)}cm")
                    break

            # *** only generate text-doc for non-full-image pages 
            if not fullpage_img:

                # *** get the text
                text = page.get_text(flags= PyMuPDF.TEXT_PRESERVE_WHITESPACE | 
                                    PyMuPDF.TEXT_INHIBIT_SPACES | 
                                    PyMuPDF.TEXT_DEHYPHENATE | 
                                    PyMuPDF.TEXT_PRESERVE_SPANS | 
                                    PyMuPDF.TEXT_MEDIABOX_CLIP, 
                                    sort=False)
                
                # *** ignore pages with less than 100 characters
                if len(text) < 80:
                    print(f"{fname} Seite:{page.number+1} ignoriert, da kein brauchbarer Text ({len(text)} Zechen)")
                else:

                    # *** tidy-up the text
                    # substitute strange characters & known ligatures
                    text = text.replace('€', 'Euro')
                    text = text.replace('„', '"')             # Anführungszeichen-Anfang
                    text = text.replace('—', '-')             # m-dash
                    text = text.replace('\'', '"')            # replace single with double quotes
                    text = text.replace('\t', ' ')            # replace tabs with a space
                    text = text.replace('\r', '')             # delete carriage returns
                    text = text.replace('\v', '')             # delete vertical tabs
                    text = text.replace('(cid:297)', 'fb')
                    text = text.replace('(cid:322)', 'fj')
                    text = text.replace('(cid:325)', 'fk')
                    text = text.replace('(cid:332)', 'ft')
                    text = text.replace('(cid:414)', 'tf')
                    text = text.replace('(cid:415)', 'ti')
                    text = text.replace('(cid:425)', 'tt')
                    text = text.replace('(cid:426)', 'ttf')
                    text = text.replace('(cid:427)', 'tti')
                    text = text.replace('•', '*')
                    text = text.replace('\uf0b7', '*')
                    text = text.replace('\uf0b0', '-')
                    text = text.replace('\uf031\uf02e', '1.')
                    text = text.replace('\uf032\uf02e', '2.')
                    text = text.replace('\uf033\uf02e', '3.')
                    text = text.replace('\uf034\uf02e', '4.')
                    text = text.replace('\uf035\uf02e', '5.')
                    text = text.replace('\uf036\uf02e', '6.')
                    text = text.replace('\uf037\uf02e', '7.')
                    text = text.replace('\uf038\uf02e', '8.')
                    text = text.replace('\uf039\uf02e', '9.')
                    text = text.replace('\uf0d8', '.nicht.')
                    text = text.replace('\uf0d9', '.und.')
                    text = text.replace('\uf0da', '.oder.')
                    text = text.replace('→', '.impliziert. (Mathematisch)')
                    text = text.replace('\uf0de', '.impliziert.')
                    text = text.replace('↔', '.äquivalent. (Mathematisch)')
                    text = text.replace('\uf0db', '.äquivalent.')
                    text = text.replace('≈','.annähernd.')
                    text = text.replace('\uf061', 'Alpha')
                    text = text.replace('β', 'Beta')
                    text = text.replace('\uf067', 'Gamma')

                    # ** some substutions are dependent on "producer"
                    producer = doc.metadata.get('producer')
                    if producer.find('Print To PDF') >= 0:
                        text = regex_sub(r'h\s\�\sp','http',text)
                        text = regex_sub(r'ma\s\�\ssch','matisch',text)
                        text = regex_sub(r'ma\s\�\sk','matik',text)
                        text = regex_sub(r'ma\s\�\son','mation',text)
                        text = regex_sub(r'unk\s\�\son','unktion',text)
                        text = regex_sub(r'ddi\s\�\son','ddition',text)
                        text = regex_sub(r'ul\s\�\splika � on','ultiplikation',text)
                        text = regex_sub(r'r\s\�\skel','rtikel',text)
                        text = regex_sub(r'a\s\�\sonal','ational',text)
                        text = regex_sub(r'li\s\�\ssche','litische',text)
                        text = regex_sub(r'olu\s\�\son','olution',text)
                        text = regex_sub(r's\s\�\smm','stimm',text)
                        text = regex_sub(r's\s\�\sge','stige',text)
                        text = regex_sub(r'a\s\�\son','ation',text)
                        text = regex_sub(r'scha\s\�\sen','schaften',text)
                        text = regex_sub(r'scha\s\�\ss','schafts',text)
                        text = regex_sub(r'a\s\�\sor','atfor',text)
                        text = regex_sub(r'\s\�\s','??',text)            # catchall

                    # ** some Substutions are independent of PDF-Generators
                    # remove line-break hyphenations
                    text = regex_sub(r'-\n', '',text)
                    # remove training spaces in lines
                    text =text.replace(' +\n', '\n')
                    # change single \n in content to " ", but not multiple \n
                    text = regex_sub(r'(?<!\n)\n(?!\n)', ' ',text)
                    # change multiple consecutive \n in content to just one \n
                    text = regex_sub(r'\n{2,}', '\n',text)
                    # remove strange single-characters with optional leading and trailing spaces in lines
                    text = regex_sub(r'\n *(\w|\*) *\n', '\n',text)
                    # remove strange single-character sequences with spaces inbetween texts
                    text = regex_sub(r'((\w|/|:) +){3,}(\w|/|:)', '',text)
                    # replace multiple blanks with just one
                    text = regex_sub(r'  +', ' ',text)

                    # *** return a Langchain_Document for each non-empty page
                    if len(text) > 0:
                        return_docs.append(Langchain_Document(
                                metadata = {**doc_metadata, **{'page':page.number+1}},
                                page_content = text,
                            ))

    return return_docs

"""
Initialize Text Splitter
"""
# dynamically import the langchain text splitter class and instantiate it
from importlib import import_module
text_splitter_module = import_module("langchain.text_splitter")
TextSplitter = getattr(text_splitter_module, text_splitter_name)
text_splitter = TextSplitter(**text_splitter_parameters)

"""
Initialize Embeddings
"""
from langchain.embeddings import HuggingFaceEmbeddings
print(f"Embeddings {embeddings_model_name} werden eingelesen...\n")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

"""
Initialize ChromaDB
"""
from langchain.vectorstores import Chroma
from chromadb.config import Settings as Chroma_Settings
# Define the Chroma settings
chroma_settings = Chroma_Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)

"""
Checks if ChromaDB exists
"""
from os import system as os_system, path as os_path
from glob import glob
def does_db_exist() -> bool:
    if os_path.exists(os_path.join(persist_directory, 'index')):
        if os_path.exists(os_path.join(persist_directory, 'chroma-collections.parquet')) and os_path.exists(os_path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob(os_path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob(os_path.join(persist_directory, 'index/*.pkl'))
            return True
    return False

"""
parse source_directory (for full import) + append_directory for all filenames to load
"""
full_import_paths = []
full_import_paths.extend(
    glob(os_path.join(source_directory, f"**/*.pdf"), recursive=True)
)
append_paths = []
append_paths.extend(
    glob(os_path.join(append_directory, f"**/*.pdf"), recursive=True)
)

"""
Load and convert file_path into a langchain document
"""
def load_file(file_path: str) -> Langchain_Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    return myPDFLoader(file_path)

"""
Load + process all documents
"""
# decide if we append to existing db or create a new one
if does_db_exist():
    if len(append_paths) > 0:
        print(f"Dokumentdateien in {append_directory} werden eingelesen und verarbeitet...\n")
        create_db = False
        file_paths = append_paths
        move_from_append = True
    else:
        print(f"Es existiert bereits eine Wissensdatenbank in {persist_directory}.\n")
        print(f"Um Dokumente hinzuzufügen, lege diese im Ordner {append_directory} ab und starte den Import erneut.\n")
        print(f"Um eine neue Wissensdatenbank anzulegen, lösche den Ordner {persist_directory} und starte den Import erneut.\n")
        exit()
else:
    print(f"{persist_directory} wird gelöscht und neu erzeugt.\n")
    os_system(f'rm -rf {persist_directory}')
    print(f"Dokumentdateien in {source_directory} werden eingelesen und verarbeitet...\n")
    create_db = True
    file_paths = full_import_paths
    move_from_append = False

# process all documents
db = None
total_pages=0
total_chunks=0
for idx,file_path in enumerate(file_paths):
    print(f"Datei {file_path} ({idx+1}/{len(file_paths)})...")
    documents=load_file(file_path)                  # txt as 1 document, pdfs as 1 document per page
    pages=len(documents)
    total_pages += pages
    print(f"... wurde eingelesen und in {pages} Seite(n) umgewandelt ...")
    # Split into chunks of text
    chunks = text_splitter.split_documents(documents) if len(documents) > 0 else []
    print(f"... zerteilt auf {len(chunks)} Textteil(e) ...")
    total_chunks += len(chunks)
    # tidying-up chunk text and numbering of chunks per source
    n_chunk=1
    chunk_metadata=chunks[0].metadata
    for i in range(len(chunks)):
        # new chunk begins if metadata is different
        if chunk_metadata!=chunks[i].metadata:
            n_chunk=1
            chunk_metadata=chunks[i].metadata
        else:
            n_chunk += 1
        chunks[i].metadata.update({"chunk": f"{n_chunk}"})

        # final tidying-up of chunk text - needed because of SpaCy weirdnesses
        txt = chunks[i].page_content
        # change single \n in content to " ", but not multiple \n
        txt = regex_sub(r'(?<!\n)\n(?!\n)', ' ',txt)
        # change multiple consecutive \n in content to just one \n
        txt = regex_sub(r'\n{2,}', '\n',txt)
        # replace txt if changed
        if txt != chunks[i].page_content:
            chunks[i].page_content = txt
               
    # create embeddings and persist
    if db is None:
        # create or append-to db
        if create_db:
            db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory, client_settings=chroma_settings)
        else:
            db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=chroma_settings)
            db.add_documents(chunks)
    else:
        # add to existing db
        db.add_documents(chunks)
    db.persist()
    print("... und in der Wissensdatenbank gespeichert.\n")

# move files from append_directory to source_directory after finishing import
if move_from_append:
    print(f"Verschiebe alle Dateien aus {append_directory} nach {source_directory}...\n")
    os_system(f'mv {append_directory}/* {source_directory}/')

# Statistics
print(f"Insgesamt {len(file_paths)} Dokument(e) mit {total_pages} Seite(n) und {total_chunks} Textteil(en) wurden eingelesen.\n")
db = None
