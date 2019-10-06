def n_grams(text,n):
    """
    takes tokens or text, returns a list of n-grams
    """

    return [text[i:i+n] for i in range(len(text) -n + 1)]


cleaned = ['mary', 'do', "n't", 'slap', 'the', 'green', 'witch', '.']
print(n_grams(cleaned,3))