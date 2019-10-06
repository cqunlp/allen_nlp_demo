import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Mary slapped the green witch.")
for token in doc:
    print("{} - {}".format(token, token.pos_))