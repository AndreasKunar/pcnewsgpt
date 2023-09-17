"""
*** PCnewsGPT Hilfsprogramm: Test PDF Import mit LLM-Analyse ***

*** WORK IN PROGRESS !!! ***

<<<<<<< HEAD
Bisherige Ergebnisse:
- die verbleibenden Ligaturlücken können vom LLM nicht geschlossen werden -> Verbesserung/Adaption PDF-importer mittels Tabelle?
- wir brauchen einen besseren Prompt fürs Zerteilen der Inhalte, ev. erkennen des Inhaltsverzeichnisses
=======
ToDo: 
1) Umstellen auf neuen PDF-Importer (Basis: import.py V1.5)
2) Analysieren der LLM Ausgabe zu den Wissensfragmenten
3) ...
>>>>>>> 17aed3113d220e857f47494a33a70189b40eec38

"""

# Central prompt template (note: substituted context has \n at end)
<<<<<<< HEAD
prompt_template = "### Aufgabe: Formuliere den folgenden Text neu:" + \
                  "'{}' ### Antwort: "
=======
prompt_template = "### Formuliere den folgenden Text neu:" + \
                  "{}\n"
>>>>>>> 17aed3113d220e857f47494a33a70189b40eec38

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ
load_dotenv()
model_path = os_environ.get('MODEL_PATH','llama-2-13b-chat.ggmlv3.q4_0.bin')
model_n_ctx = int(os_environ.get('MODEL_N_CTX',2048))
model_temp = float(os_environ.get('MODEL_TEMP',0.4))
max_tokens = int(os_environ.get('MAX_TOKENS',500))
model_threads = int(os_environ.get('MODEL_THREADS',8))
model_n_gpu = int(os_environ.get('MODEL_GPU',1))
source_directory = os_environ.get('SOURCE_DIRECTORY','source_documents')

<<<<<<< HEAD
=======
# PDFLoader
from langchain.document_loaders import (
    PyMuPDFLoader,
)
loader_class = PyMuPDFLoader
loader_args = {}

>>>>>>> 17aed3113d220e857f47494a33a70189b40eec38
"""
Initial banner Message
"""
print("\nPCnewsGPT PDF-IMport mit LLM Analyse V0.0.1\n")

"""
Initialize LLM
"""
print(f"KI-Model {model_path} wird geladen...\n")
from llama_cpp import Llama
llm = Llama(model_path=model_path,
            n_ctx=int(model_n_ctx*1.5),
            logits_all=False, 
            embedding = False,
            n_threads = model_threads,
            n_gpu_layers = model_n_gpu,
            verbose = False,
        )
<<<<<<< HEAD

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
=======
"""

Load and convert file_path into a langchain document
"""
from langchain.docstore.document import Document as langchain_Document
from re import sub as regex_sub
def load_file(file_path: str, loader_class, loader_args) -> langchain_Document:
    # load via langchain
    loader = loader_class(file_path, **loader_args)
    lc_doc = loader.load()
    for i in range(len(lc_doc)):
        # Tidy up PDFs
        doc = lc_doc[i].page_content
        source = lc_doc[i].metadata['source']
        page_num = lc_doc[i].metadata.get('page', None)
        creationdate = lc_doc[i].metadata.get('creationdate', None)
        # remove line-break hyphenations
        doc = regex_sub(r'-\n+ *', '',doc)
        # remove training spaces in lines
        doc =doc.replace(' \n', '\n')
        # remove excess spaces
        doc = doc.replace('  ', ' ')
        # substitute known ligatures & strange characters
        doc = doc.replace('(cid:297)', 'fb')
        doc = doc.replace('(cid:322)', 'fj')
        doc = doc.replace('(cid:325)', 'fk')
        doc = doc.replace('(cid:332)', 'ft')
        doc = doc.replace('(cid:414)', 'tf')
        doc = doc.replace('(cid:415)', 'ti')
        doc = doc.replace('(cid:425)', 'tt')
        doc = doc.replace('(cid:426)', 'ttf')
        doc = doc.replace('(cid:427)', 'tti')
        doc = doc.replace('\uf0b7', '*')
        doc = doc.replace('•', '*')
        doc = doc.replace('\uf031\uf02e', '1.')
        doc = doc.replace('\uf032\uf02e', '2.')
        doc = doc.replace('\uf033\uf02e', '3.')
        doc = doc.replace('\uf034\uf02e', '4.')
        doc = doc.replace('\uf035\uf02e', '5.')
        doc = doc.replace('\uf036\uf02e', '6.')
        doc = doc.replace('\uf037\uf02e', '7.')
        doc = doc.replace('\uf038\uf02e', '8.')
        doc = doc.replace('\uf039\uf02e', '9.')
        doc = doc.replace('\uf0d8', '.nicht.')
        doc = doc.replace('\uf0d9', '.und.')
        doc = doc.replace('\uf0da', '.oder.')
        doc = doc.replace('→', '.impliziert. (Mathematisch)')
        doc = doc.replace('\uf0de', '.impliziert.')
        doc = doc.replace('↔', '.äquivalent. (Mathematisch)')
        doc = doc.replace('\uf0db', '.äquivalent.')
        doc = doc.replace('≈','.annähernd.')
        doc = doc.replace('\uf061', 'Alpha')
        doc = doc.replace('β', 'Beta')
        doc = doc.replace('\uf067', 'Gamma')
        # substiture other strange characters
        doc = doc.replace('€', 'Euro')
        doc = doc.replace("„", '"')             # Anführungszeichen
        doc = doc.replace("—", '"')             # m-dash
        doc = doc.replace("'", '"')             # replace single with double quotes
        doc = doc.replace("\t", " ")            # replace tabs with a space
        doc = doc.replace("\r", "")             # delete carriage returns
        doc = doc.replace("\v", "")             # delete vertical tabs
        
        # change single \n in content to " ", but not multiple \n
        doc = regex_sub(r'(?<!\n)\n(?!\n)', ' ',doc)
        # change multiple consecutive \n in content to just one \n
        doc = regex_sub(r'\n{2,}', '\n',doc)
        # remove strange single-characters with optional leading and trailing spaces in lines
        doc = regex_sub(r'\n *(\w|\*) *\n', '\n',doc)
        # remove strange single-characters with spaces inbetween texts
        doc = regex_sub(r'((\w|/|:) +){3,}(\w|/|:)', '',doc)
        # remove multiple blanks
        doc = regex_sub(r'  +', ' ',doc)
        
        # split doc-content into pages and remove any trailing empty pages
        pages =doc.split('\x0c')
        while (len(pages) > 1) and (pages[-1] == ''):
            pages.pop()
        # if there are no remaining pages, empty the text in lc_doc
        if len(pages) == 0:
            lc_doc[i].page_content = ""
        # if its just one page, update lc_doc with processed content
        elif len(pages) == 1:
            lc_doc[i].page_content = pages[0]
            lc_doc[i].metadata = {"source": f"{source}"}
            if page_num is not None:
                lc_doc[i].metadata.update({"page": f"{page_num+1}"})
            if creationdate is not None:
                lc_doc[i].metadata.update({"creationdate": f"'{regex_sub(r'[^0-9]','',creationdate)[:8]}'" })
        # if there are muliple pages, create new lc_doc from non-empty pages
        else:
            del lc_doc[i]    # we need a new name + content as multiple pages
            pg_num=0
            for page in pages:
                if page != '':    # only add non-empty pages, keep page numbering
                    metadata = {"source": f"{source}", "page": f"{pg_num+1}"}
                    if creationdate is not None:
                        metadata.update({"creationdate": f"'{regex_sub(r'[^0-9]','',creationdate)[:8]}'" })
                    lc_doc.insert(i, langchain_Document(
                        page_content = page,
                        metadata=metadata
                    ))
                    i+=1
                pg_num += 1

    # remove empty documents
    docs=[]
    for i in range(len(lc_doc)):
        if lc_doc[i].page_content != '':
            docs.append(lc_doc[i])
    # return the loaded doc               
    return docs

>>>>>>> 17aed3113d220e857f47494a33a70189b40eec38

"""
process all files in import directory
"""
from glob import glob
from os import path as os_path
for file_path in glob(os_path.join(source_directory, f"**/*.pdf"), recursive=True):
    # load a file
<<<<<<< HEAD
    document=myPDFLoader(file_path)

    # process its pages
    for page in document:
        # print page metadata
        print(f"('metadata':{page.metadata},")

        # analyze content with LLM
        prompt = prompt_template.format(page.page_content)
        p=prompt.replace('\n','\\n')
        print(f"'prompt':'{p}',",flush=True)
=======
    document=load_file(file_path, loader_class, loader_args)

    # process its pages
    for page in document:
        # print paage it data
        print(f"('metadata':{page.metadata},\n  'page_content':'",end="",flush=True)

        # analyze content with LLM
        prompt = prompt_template.format(page.page_content)    
>>>>>>> 17aed3113d220e857f47494a33a70189b40eec38
        llm_result = llm(
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = model_temp,
            echo = False,
            stop=["###"],
            stream = False,
            )
<<<<<<< HEAD
        print(f"'page_content':'{llm_result.get('choices')[0].get('text').strip()}'",end="")
=======
        print(f"{llm_result.get('choices')[0].get('text').strip()}'",end="")
>>>>>>> 17aed3113d220e857f47494a33a70189b40eec38
        if int(llm_result.get('usage').get('total_tokens')) > model_n_ctx:
            print(f",'error_size_to_large':{llm_result.get('usage').get('total_tokens')}",end="")
        print("),")
        llm.reset()
