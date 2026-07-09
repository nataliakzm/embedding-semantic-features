from data.CS.csense1 import extended_spairs as csense1_sentences
from data.CS.csense0 import extended_spairs as csense0_sentences

# Build pairs programmatically: (SentA from csense1, SentB from csense0)
pairs = [
    (sent_a, sent_b)
    for sent_a, sent_b in zip(csense1_sentences, csense0_sentences)
]

category = ', '.join([f'{sentence[0]}, {sentence[1]}' for sentence in pairs])

extended_spairs = [
    sentence
    for pair in pairs
    for sentence in pair
]