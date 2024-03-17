import random

import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import xlrd
from embed import encode_and_reduce
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

AAgroup = {"C":["C"],
      "A":["G","P","S","T"],
      "G":["A","P","S","T"],
      "P":["G","A","S","T"],
      "S":["G","P","A","T"],
      "T":["G","P","S","A"],
      "N":["D","Q","E"],
      "D":["N","Q","E"],
      "Q":["D","N","E"],
      "E":["D","Q","N"],
      "R":["H","K"],
      "H":["R","K"],
      "K":["H","R"],
      "I":["L","M","V"],
      "L":["I","M","V"],
      "M":["L","I","V"],
      "V":["L","M","I"],
      "F":["W","Y"],
      "W":["F","Y"],
      "Y":["W","F"]
      }

hydro = pd.read_excel(f"data/hydrophobic.xlsx", index_col=0)
for i in range(len(hydro.columns)):
    hydro.iat[i, i] = 0
hydro = hydro.to_dict()

#Sites with the highest conservation score  60
flu_sites = {
"HA" : [160, 158, 176, 137, 151, 156, 108, 294, 19, 209, 241, 78, 175, 161, 138, 172, 189, 69, 239, 327, 9, 66, 94, 242, 505, 277, 64, 110, 214, 47, 174, 147, 179, 228, 292, 153, 187, 202, 14, 61, 208, 49, 546, 205, 91, 188, 243, 315, 149, 99, 422, 173, 144, 16, 22, 276, 377, 73, 328, 15],
"M1" : [227, 142, 95, 121, 15, 167, 166, 218, 239, 115, 230, 205, 137, 101, 174, 248, 224, 107, 207, 219, 144, 140, 181, 234, 157, 37, 139, 242, 46, 30, 116, 168, 209, 214, 232, 27, 59, 235, 213, 231, 147, 200, 36, 192, 211, 208, 222, 98, 126, 85, 246, 125, 41, 33, 80, 160, 225, 91, 82, 249],
"NA" : [93, 267, 42, 81, 40, 369, 221, 435, 385, 339, 307, 143, 43, 380, 402, 392, 52, 344, 199, 336, 310, 381, 215, 44, 45, 51, 346, 338, 329, 313, 82, 265, 216, 399, 150, 400, 468, 126, 245, 372, 41, 464, 147, 387, 386, 258, 332, 249, 62, 172, 367, 370, 23, 416, 83, 210, 384, 56, 30, 308],
"NP" :[353, 373, 450, 217, 34, 52, 136, 375, 100, 406, 384, 423, 305, 109, 105, 313, 357, 350, 452, 377, 351, 433, 33, 371, 16, 61, 186, 31, 286, 343, 239, 455, 293, 214, 372, 283, 280, 18, 344, 442, 422, 459, 312, 65, 131, 430, 189, 197, 456, 190, 498, 21, 119, 425, 289, 400, 77, 473, 444, 38],
"NS1": [56, 171, 59, 26, 67, 84, 226, 60, 129, 22, 112, 221, 101, 23, 28, 55, 166, 98, 21, 114, 71, 139, 227, 215, 95, 127, 125, 209, 81, 48, 197, 70, 144, 135, 82, 206, 196, 41, 145, 90, 111, 211, 18, 76, 205, 44, 137, 207, 194, 108, 27, 25, 80, 73, 63, 87, 152, 116, 225, 204],
"PA" : [272, 400, 28, 321, 337, 101, 388, 256, 277, 421, 618, 142, 100, 65, 343, 66, 57, 184, 404, 332, 387, 269, 323, 62, 396, 383, 225, 557, 268, 356, 552, 204, 311, 55, 385, 573, 409, 675, 382, 407, 668, 348, 716, 391, 118, 61, 254, 362, 63, 626, 423, 554, 403, 684, 497, 607, 208, 405, 262, 545],
"PB1": [113, 212, 361, 584, 216, 741, 52, 581, 621, 375, 336, 619, 486, 430, 587, 576, 327, 709, 586, 179, 638, 339, 642, 433, 59, 383, 387, 215, 374, 172, 578, 14, 628, 257, 368, 386, 667, 573, 48, 54, 694, 213, 398, 384, 618, 211, 171, 654, 397, 177, 317, 181, 737, 57, 517, 200, 214, 111, 745, 156],
"PB2" : [559, 588, 676, 292, 613, 453, 81, 64, 249, 105, 82, 674, 451, 271, 368, 194, 456, 590, 684, 661, 9, 340, 667, 627, 199, 463, 120, 702, 567, 682, 44, 382, 475, 227, 67, 569, 526, 697, 461, 353, 560, 147, 478, 65, 591, 645, 391, 473, 299, 575, 63, 598, 61, 389, 338, 255, 677, 106, 80, 648]
}
spike_sites = [526,1248,657,857,2145,1500,671,10,283,289,2165,1606,378,1508,1497,1181,608,1192,762,1495,1230,843,709,1012,9,1386,1253,287,1233,1014,1377,467,888,858,915,1572,1489,1498,1579,341,1488,1191,822,1164,1499,428,529,1877,682,626,656,323,425,1335,1251,861,1356,2202,913,489]



def seqAug(seq, indexes=None, method="conservation", n_res=15):
    result_str = list(seq)
    distance_dict = {
        "conservation": AAgroup,
        "hydrophobic": hydro,
        "random": list("ACDEFGHIKLMNPQRSTVWY")
    }[method]

    if indexes is None:
        indexes = random.sample(range(len(seq)), n_res)

    for index in indexes[:n_res]:
        if index<0 or index > len(seq)-1 : continue
        char_to_replace = seq[index]
        if char_to_replace not in distance_dict:
            continue

        if method == "conservation":
            valid_letters = distance_dict[char_to_replace]
        elif method == "hydrophobic":
            valid_letters = [key for key, value in distance_dict.items() if value >= 8 and key != char_to_replace]
        else:  # method == "random"
            valid_letters = distance_dict

        replacement_char = random.choice(valid_letters) if valid_letters else char_to_replace
        result_str[index] = replacement_char

    return ''.join(result_str)



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



def flu_dataloader(type,batch_size,n_res,test_size=0.2):
    fasta_file = f'data/FluCnn/{type}.fasta'
    sequences = []
    aug_seqs = []
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
        host_info = record.description.split('|')[-1].strip()

        label = host_mapping.get(host_info, -1)
        if label != -1:

            sequences.append(sequence)
            # sequence augmentation
            aug_seq = seqAug(sequence,flu_sites[type],method="conservation",n_res=n_res)
            aug_seqs.append(aug_seq)
            labels.append(label)


    encoded_sequences = encode_and_reduce(sequences, ["one_hot_encode"])
    encoded_aug_seq = encode_and_reduce(aug_seqs, ["one_hot_encode"])
    encoded_sequences = [list(pair) for pair in zip(encoded_sequences, encoded_aug_seq)]

    X_train, X_test, y_train, y_test = train_test_split(encoded_sequences, labels, test_size=test_size, random_state=42)

    train_dataset = setDataset(X_train, y_train)
    test_dataset = setDataset(X_test, y_test)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False,pin_memory=True, num_workers=8)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=4)
    return train_dataset,test_dataset


def cov_dataloader(batch_size,n_res,test_size=0.2):
    fasta_file = "data/CoVs/Sequences.fasta"
    sequences_dictionary = {sequence.id: sequence.seq for sequence in SeqIO.parse(fasta_file, 'fasta')}
    deflines = [entry for entry in sequences_dictionary.keys()]  # create a list of deflines
    sequences = [entry for entry in sequences_dictionary.values()]
    # sequence augmentation
    aug_seqs = [seqAug(sequence,spike_sites,method="conservation",n_res=n_res) for sequence in sequences]
    sequences = encode_and_reduce(sequences,["one_hot_encode"])
    aug_seqs = encode_and_reduce(aug_seqs,["one_hot_encode"])
    sequences = [list(pair) for pair in zip(sequences, aug_seqs)]

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False,pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False,pin_memory=True, num_workers=4)
    return train_loader,val_loader

def vir_dataloader(type,batch_size,n_res,test_size=0.2):
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

    # sequence augmentation
    aug_seqs = [seqAug(sequence, flu_sites[type], method="conservation", n_res=n_res) for sequence in raw_seq]
    raw_seq = encode_and_reduce(raw_seq,["protvec"])
    aug_seqs = encode_and_reduce(aug_seqs,["protvec"])
    sequences = [list(pair) for pair in zip(raw_seq, aug_seqs)]

    labels = raw_label
    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=test_size, random_state=42)
    train_dataset = setDataset(X_train, y_train)
    val_dataset = setDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False,pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=4)
    return train_loader,val_loader

def anti_dataloader(type,batch_size,n_res,test_size=0.2):
    antigenic_dist = pd.read_csv(f'data/IAVCnn/{type}_antigenic.csv')
    seq = pd.read_csv(f'data/IAVCnn/{type}_sequence_HA1.csv', names=['seq', 'description'])
    raw_data = strain_selection(antigenic_dist, seq,n_res)
    labels = raw_data[4]
    feature1 = encode_and_reduce(raw_data[0], ["protvec"])
    feature2 = encode_and_reduce(raw_data[1], ["protvec"])
    feature3 = encode_and_reduce(raw_data[2], ["protvec"])
    feature4 = encode_and_reduce(raw_data[3], ["protvec"])

    feature1 = np.array(feature1) - np.array(feature2)
    feature2 = np.array(feature3) - np.array(feature4)
    feature = [list(pair) for pair in zip(feature1, feature2)]

    X_train, X_val, y_train, y_val = train_test_split(feature, labels, test_size=test_size, random_state=42)
    train_dataset = setDataset(X_train, y_train)
    val_dataset = setDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=False,pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=4)

    return train_loader, val_loader

def strain_selection(distance_data, seq_data,n_res):
    raw_data = []
    strain1 = []
    strain2 = []
    strain3 = []
    strain4 = []
    label = calculate_label(distance_data)
    for i in range(0, distance_data.shape[0]):

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
        # sequence augmentation
        strain3.append(seqAug(seq1, np.array(flu_sites["HA"])-16, method="conservation", n_res=n_res))
        strain4.append(seqAug(seq2, np.array(flu_sites["HA"])-16, method="conservation", n_res=n_res))

    raw_data.append(strain1)
    raw_data.append(strain2)

    raw_data.append(strain3)
    raw_data.append(strain4)
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
