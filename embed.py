import random
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler


pv = pd.read_csv('data/protVec_100d_3grams.csv', delimiter='\t')
trigrams = list(pv['words'])
trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
trigram_vecs = pv.loc[:, pv.columns != 'words'].values

def encode_amino_acids(sequences, encoding_methods):
   
    amino_acid_encoding = {
        'one_hot_encode': one_hot_encode,
        'one_hot_position': one_hot_position,
        'protvec': protvec,
    }
    Btworandom = 'DN'
    Jtworandom = 'IL'
    Ztworandom = 'EQ'
    Xallrandom = 'ACDEFGHIKLMNPQRSTVWY'

    encoded_sequence = []
    for sequence in sequences:
        if isinstance(sequence, list):
            sequence = ''.join(sequence)
        sequence = sequence.replace('B', random.choice(Btworandom))
        sequence = sequence.replace('J', random.choice(Jtworandom))
        sequence = sequence.replace('Z', random.choice(Ztworandom))
        sequence = sequence.replace('X', random.choice(Xallrandom))
      
        aa_encoding = []
        for method in encoding_methods:
            aa_encoding.extend(amino_acid_encoding[method](sequence))
        encoded_sequence.append(aa_encoding)

    return encoded_sequence


def encode_and_reduce(sequence,encoding_methods, use_pca=False, n_components=2):

    sequence_features = encode_amino_acids(sequence, encoding_methods)

    if use_pca:
        scaler = StandardScaler()
        sequence_features_scaled = scaler.fit_transform(sequence_features)

     
        pca = PCA(n_components=n_components)
        reduced_sequence_features = pca.fit_transform(sequence_features_scaled)
        return reduced_sequence_features
    else:
      
        return sequence_features

def one_hot_encode(sequence):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.zeros((len(sequence), len(alphabet)))

    for i, aa in enumerate(sequence):
        if aa in alphabet:
            encoding[i, alphabet.index(aa)] = 1
    return encoding

def one_hot_position(sequence):
    l = len(sequence)-1
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    encoding = np.zeros((len(sequence), len(alphabet)))

    for i, aa in enumerate(sequence):
        if aa in alphabet:
            encoding[i, alphabet.index(aa)*l+i] = 1
    return encoding




def protvec(sequence):

    l = len(sequence)

    # embedding with ProVect

    strain_embedding = []
    # overlapped feature generation for 3-grams
    for i in range(0, l - 2):
        trigram = sequence[i:i + 3]
        if trigram[0] == '-' or trigram[1] == '-' or trigram[2] == '-':
            tri_embedding = trigram_vecs[trigram_to_idx['<unk>']]
        else:
            tri_embedding = trigram_vecs[trigram_to_idx["".join(trigram)]]

        strain_embedding.append(tri_embedding)

    return strain_embedding

