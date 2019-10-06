import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print("{} - {}".format(chunk, chunk.label_))