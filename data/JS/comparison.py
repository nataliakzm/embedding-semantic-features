from data.JS.jsense1 import extended_spairs as jsense1_sentences
from data.JS.jsense0 import extended_spairs as jsense0_sentences

# Build pairs programmatically: (SentA from jsense1, SentB from jsense0)
pairs = [
    (sent_a, sent_b)
    for sent_a, sent_b in zip(jsense0_sentences, jsense1_sentences)
]

category = ', '.join([f'{sentence[0]}, {sentence[1]}' for sentence in pairs])

extended_spairs = [
    sentence
    for pair in pairs
    for sentence in pair
]
