# !pip install PyMuPDF
def pdf_to_text(src):

    import fitz
    import re
    import unicodedata

    def extract_useful_text_from_pdf(src):

        def is_useful(text):
            return not re.match(r"^(Figure|Table)\s+(\d+):", text)

        useful_text = []
        doc = fitz.open(src)

        for page in doc:
            for block in page.get_text("dict")["blocks"]:

                if block['type'] == 0:

                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += unicodedata.normalize("NFKD", span["text"]) + " "
                    
                    if is_useful(text):
                        useful_text.append(text)
        
        doc.close()
        return useful_text


    extracted_text = extract_useful_text_from_pdf(src)

    toc = ['_'] + [unicodedata.normalize("NFKD", i[1]).strip() for i in fitz.open(src).get_toc()]

    cnt = 0
    maxlen = len(toc)-1

    res = []
    buf = ""
    for text in extracted_text:

        if cnt < maxlen and text.strip() == toc[cnt+1]:
            if buf:
                res += [{'section': toc[cnt], 'text': buf}]
                buf = ""
            cnt += 1
        
        else:
            buf += text

    res += [{'section': toc[cnt], 'text': buf}]
    return res




def pdf_to_text_spans(src):

    import fitz
    import re
    import unicodedata

    def extract_useful_text_from_pdf(src):

        def is_useful(text):
            return not re.match(r"^(Figure|Table)\s+(\d+):", text)
        
        abstract = None
        def is_section(span):
            return (span["size"], span["flags"]) == abstract

        useful_text = []
        doc = fitz.open(src)

        toc_span = []

        for page in doc:
            for block in page.get_text("dict")["blocks"]:

                if block['type'] == 0:
                    section = ""
                    text = ""
                    for line in block["lines"]:

                        for span in line["spans"]:
                            norm_text = unicodedata.normalize("NFKD", span["text"])

                            if not abstract and norm_text.strip() == "Abstract":
                                abstract = (span["size"], span["flags"])
                            
                            if abstract and is_section(span):
                                section += span["text"] + " "
                            
                            text += norm_text + " "

                    if is_useful(text):
                        useful_text.append(text)
                    
                    if section: toc_span += [section.strip()]

        doc.close()
        return useful_text, toc_span


    extracted_text, toc = extract_useful_text_from_pdf(src)
    cnt = 0
    maxlen = len(toc)-1

    res = []
    buf = ""
    for text in extracted_text:

        if cnt < maxlen and text.strip() == toc[cnt+1]:
            if buf:
                res += [{'section': toc[cnt], 'text': buf}]
                buf = ""
            cnt += 1
        
        else:
            buf += text

    res += [{'section': toc[cnt], 'text': buf}]
    return res