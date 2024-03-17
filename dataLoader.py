import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import xlrd
from embed import encode_and_reduce
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class setDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return torch.Tensor(sequence), torch.LongTensor([label])

def flu_dataloader(type,batch_size,test_size=0.2):
  
    fasta_file = f'data/FluCnn/{type}.fasta'
    sequences = []
    labels = []
    host_mapping = {
        "Avian": 0,
        "Human": 1,
        "Swine": 2,
        "Environment": 3,
        "Canine": 4
       
    }
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        host_info = record.description.split('|')[-1].strip()  # Assuming host info is the last part of the description.

        label = host_mapping.get(host_info, -1)
        if label != -1:
            sequences.append(sequence)
            labels.append(label)

    encoded_sequences = encode_and_reduce(sequences,["one_hot_encode"])

    X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, test_size=test_size, random_state=42)

    train_dataset = setDataset(X_train, y_train)
    test_dataset = setDataset(X_test, y_test)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, drop_last=True,shuffle=False)
    return train_dataset,test_dataset


def cov_dataloader(batch_size,test_size=0.2):
    fasta_file = "data/CoVs/Sequences.fasta"
    sequences_dictionary = {sequence.id: sequence.seq for sequence in SeqIO.parse(fasta_file, 'fasta')}
    deflines = [entry for entry in sequences_dictionary.keys()]  # create a list of deflines
    protein_sequences = [entry for entry in sequences_dictionary.values()]

    sequences = encode_and_reduce(protein_sequences,["one_hot_encode"])

    labels = []
    for i in range(0, len(deflines)):
        host_species = deflines[i].split("|")[2]
        labels.append(host_species)
    hosts_dictionary = {
        "Avians": {'Turkey', 'White_Rumped_Munia', 'Bulbul', 'Pheasant', 'Night_Heron', 'Teal', 'White_Eye',
                   'White_eye', 'Duck', 'Dabbling_Duck', 'Pigeon', 'Chicken', 'Sparrow', 'Common_Moorhen',
                   'Common_moorhen',
                   'Magpie_Robin', 'Magpie_robin', 'Quail', 'Falcon', 'Bustard', 'Shelduck', 'Guinea_Fowl',
                   'Grey_Backed_Thrush'},
        "Fish": {'Blicca_Bjoerkna_L.', 'Fathead_Minnow', 'Fathead_minnow'},
        "Reptiles": {'Python', 'Turtle'},
        "Bats": {'Bat', 'Rhinolophus_blasii', 'bat_BF_258I', 'bat_BF_506I'},
        "Camels": {'Camel'},
        "Human": {'Human'},
        "Swine": {'Swine', 'Pig', 'Sus_scrofa_domesticus'},
        "Other mammals": {'Anteater', 'Hedgehog', 'Horse', 'Mink', 'Alpaca', 'Ferret', 'Rabbit',
                          'Buffalo', 'Goat', 'Chimpanzee', 'Dog', 'Cattle', 'Rat', 'Mouse', 'Cat',
                          'Dolphin',
                          'Mus_Musculus__Severe_Combined_Immunedeficiency__Scid___Female__6_8_Weeks_Old__Liver__Sample_Id:_E4m31',
                          },
        "Unknowns": {'NA', 'Unknown'}
    }
    ls = []
    for i in range(len(labels)):
        for category, set_of_hosts in hosts_dictionary.items():
            if labels[i] in set_of_hosts:
                ls.append(category)
    host_mapping = {
        "Human": 0,
        "Bats": 1,
        "Camels": 2,
        "Avians": 3,
        "Swine": 4,
        "Other mammals": 5
    }
    labels = []
    for label in ls:
        Target = host_mapping.get(label, -1)
        if label != -1:
            labels.append(Target)
        else:
            raise ValueError("wrong label")

    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_size, random_state=42)
    train_dataset = setDataset(X_train, y_train)
    val_dataset = setDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
    return train_loader,val_loader

def vir_dataloader(type,batch_size,test_size=0.2):
    HA_data = xlrd.open_workbook(f"data/Virpre/{type}_data_with_site.xlsx")
    HA_data = HA_data.sheet_by_index(0)
    raw_data = []
    raw_seq = []
    raw_label = []
    for i in range(HA_data.nrows):  # print each row
        raw_data.append(HA_data.row_values(i))  # read the information
    for j in range(1, HA_data.nrows):
        raw_seq.append(raw_data[j][5:HA_data.ncols])
    for m in range(1, HA_data.nrows):
        raw_label.append(raw_data[m][3])  # 2/3 classes

    for i in range(0, len(raw_seq)):

        if raw_label[i] == 'Avirulent':
            raw_label[i] = 0
        elif raw_label[i] == 'Virulent':
            raw_label[i] = 1
        else:
            print('error')
    feature = encode_and_reduce(raw_seq,["protvec"])

    labels = raw_label
    X_train, X_val, y_train, y_val = train_test_split(feature, labels, test_size=test_size, random_state=42)
    train_dataset = setDataset(X_train, y_train)
    val_dataset = setDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,val_loader

def anti_dataloader(type,batch_size,test_size=0.2):
    antigenic_dist = pd.read_csv(f'data/IAVCnn/{type}_antigenic.csv')
    seq = pd.read_csv(f'data/IAVCnn/{type}_sequence_HA1.csv', names=['seq', 'description'])
    raw_data = strain_selection(antigenic_dist, seq)
    labels = raw_data[2]
    feature1 = encode_and_reduce(raw_data[0], ["protvec"])
    feature2 = encode_and_reduce(raw_data[1], ["protvec"])
    feature = np.array(feature1)-np.array(feature2)
    X_train, X_val, y_train, y_val = train_test_split(feature, labels, test_size=test_size, random_state=42)
    train_dataset = setDataset(X_train, y_train)
    val_dataset = setDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def strain_selection(distance_data, seq_data):
    raw_data = []
    strain1 = []
    strain2 = []
    label = calculate_label(distance_data)
    for i in range(0, distance_data.shape[0]):
        seq1 = []
        seq222 = []
        flag1 = 0
        flag2 = 0
        for j in range(0, seq_data.shape[0]):
            if str(seq_data['description'].iloc[j]).upper() == str(distance_data['Strain1'].iloc[i]).upper():
                seq1 = str(seq_data['seq'].iloc[j]).upper()
                flag1 = 1
            if str(seq_data['description'].iloc[j]).upper() == str(distance_data['Strain2'].iloc[i]).upper():
                seq2 = str(seq_data['seq'].iloc[j]).upper()
                flag2 = 1
            if flag1 == 1 and flag2 == 1:
                break
        strain1.append(seq1)
        strain2.append(seq2)

    raw_data.append(strain1)
    raw_data.append(strain2)
    raw_data.append(label)
    return raw_data

def calculate_label(antigenic_data):
    distance_label = []
    if len(set(antigenic_data['Distance'])) == 2:
        for i in range(0, antigenic_data.shape[0]):
            if antigenic_data['Distance'].iloc[i] == 1:
                distance_label.append(1)
            elif antigenic_data['Distance'].iloc[i] == 0:
                distance_label.append(0)
            else:
                print('error')
    else:
        for i in range(0, antigenic_data.shape[0]):
            if antigenic_data['Distance'].iloc[i] >= 4.0:
                distance_label.append(1)
            else:
                distance_label.append(0)
    return distance_label
