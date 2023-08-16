import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statistics import pstdev, mean
import os
import json
import time


# Uvozimo podatke s pandasom, shranimo kot okuzeni, naredimo podtabelo okuzeni_mestne
okuzeni = pd.read_csv("C:\\Users\\User\\OneDrive - Univerza v Ljubljani\\Faks\\Magisterij\\NSU\\NSU23_DN5_Part2\\okuzeni.csv")
okuzeni_mestne = okuzeni[['ljubljana', 'maribor', 'kranj', 'koper', 'celje', 'novo_mesto', 'velenje', 'nova_gorica', 'krško', 'ptuj', 'murska_sobota', 'slovenj_gradec']]
okuzeni_mestne["slovenj_gradec"].fillna(0, inplace=True)

#Določimo več hiperparametrov kot konstante. Zaenkrat...
data_len = len(okuzeni_mestne)
# število epohov
EPOCHS = 10 
# learning rate faktor
LR = 0.001
# število celic v notranji plasti nevronske mreže
HIDDEN = 32
# dolžina zaporedij dni, na katerih se model uči
SEQ_LEN = 60
# število napovednih dni (samo 7 ali pa 30 za našo nalogo)
M = 30
# m = 30
# število dni za testno in učno množico
TEST_SIZE = 300
TRAIN_SIZE = data_len - TEST_SIZE
# metoda normiranja, kakšen seznam zajamemo za mean in std
NORMIRANJE = 'po_sekvencah'
# metoda ki jo bomo uporabili
METODA = 'LSTM'
# slovar metod
#metode = {'RNN': RNN}
# mapa kjer se shranjujejo predictionsi
SAVE_PATH = r'''C:\\Users\\User\\OneDrive - Univerza v Ljubljani\\Faks\\Magisterij\\NSU\\NSU23_DN5_Part2\\predictions\\'''
# seznam mestnih občin
mestne_obcine = ['ljubljana', 'maribor', 'kranj', 'koper', 'novo_mesto', 'velenje', 'nova_gorica', 'celje', 'krško', 'ptuj', 'murska_sobota', 'slovenj_gradec']


# definiramo nevronske mreže, ki jih bomo uporabili
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x.view(len(x), 1, -1))
        x = self.fc(x.view(len(x), -1))
        return x[-1]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x.view(len(x), 1, -1))
        x = self.fc(x.view(len(x), -1))
        return x[-1]

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x.view(len(x), 1, -1))
        x = self.fc(x.view(len(x), -1))
        return x[-1]

# definiramo funkcijo, ki nam naredi ustrezne sekvence

def create_sequences(data, seq_length=SEQ_LEN):
    sequences = []
    for i in range(len(data)-seq_length):
        sequences.append((data[i:i+seq_length], data[i+seq_length:i+seq_length+1]))
    return sequences


# Naredimo funkcijo, ki nam shrani rezultate v json file, z vsemi potrebnimi informacijami. Dodamo potem še funkcijo za dostop informacij iz file-a.

def save_results(actual_predictions, epochs = EPOCHS, lr = LR, hidden = HIDDEN, seq_len = SEQ_LEN, m = M, test_size = TEST_SIZE, normiranje = NORMIRANJE, metoda = METODA, save_path=SAVE_PATH):
    file_name = metoda
    file_name += '_' + str(epochs)
    file_name += '_' + str(lr)
    file_name += '_' + str(hidden)
    file_name += '_' + str(seq_len)
    file_name += '_' + str(m)
    file_name += '_' + str(test_size)
    file_name += '_' + str(normiranje)
    file_name += '.json'
    filepath = os.path.join(save_path, file_name)
    actual_predictions["metoda"] = metoda
    actual_predictions["epochs"] = epochs
    actual_predictions["lr"] = lr
    actual_predictions["hidden"] = hidden
    actual_predictions["seq_len"] = seq_len
    actual_predictions["m"] = m
    actual_predictions["test_size"] = test_size
    actual_predictions["normiranje"] = normiranje
    actual_predictions = json.dumps(actual_predictions, indent = 4)
    with open(filepath, "w", encoding='utf-8') as file:
        file.write(actual_predictions)    

def make_name(slovarji):
    epochs = slovarji["epochs"]
    lr = slovarji["lr"]
    hidden = slovarji["hidden"]
    seq_len = slovarji["seq_len"]
    m = slovarji["m"]
    test_size = slovarji["test_size"]
    normiranje = slovarji["normiranje"]
    metoda = slovarji["metoda"]
    file_name = metoda
    file_name += '_' + str(epochs)
    file_name += '_' + str(lr)
    file_name += '_' + str(hidden)
    file_name += '_' + str(seq_len)
    file_name += '_' + str(m)
    file_name += '_' + str(test_size)
    file_name += '_' + str(normiranje)
    return file_name

# Normiranje glede na nek glavni seznam, isto odnormiranje. Kličeš dvakrat isti seznam če hočeš seznam normirat.
def normiraj(sez, glavni_sez):
    if type(glavni_sez) == list:
        glavni_sez = torch.FloatTensor(glavni_sez)
    if type(sez) == list:
        sez = torch.FloatTensor(sez)
    return (sez - glavni_sez.mean())/max(glavni_sez.std(), 1)

def odnormiraj(sez, glavni_sez):
    return max(glavni_sez.std(), 1)*sez + glavni_sez.mean()


# Funkcija, ki na posamezni občini nauči model in vrne actual predictions, se pravi ...
def fit_model(obcina, epochs = EPOCHS, lr = LR, hidden = HIDDEN, seq_len = SEQ_LEN, m = M, test_size = TEST_SIZE, normiranje = NORMIRANJE, metoda = METODA):
    y = torch.tensor(okuzeni_mestne[obcina]).float()
    train_data = y[:-test_size]
    test_data = y[-test_size:]
    train_sequences = create_sequences(train_data, seq_len)
    
    # Ni šlo drugače zaenkrat pri modelu iz slovarja.
    model = LSTM(1, hidden, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        for i, (seq, labels) in enumerate(train_sequences):
            optimizer.zero_grad()
            model.hidden = (torch.zeros(1,1,model.hidden_size))
            y_pred = model(normiraj(seq, seq))
            loss = criterion(y_pred, normiraj(labels,seq))
            loss.backward()
            optimizer.step()
            #if i%100 == 0:
            #    print(f'i: {i:03d}, Loss: {loss:.4f}')

    train_size = data_len - test_size
    test_inputs = y[-seq_len + train_size:].tolist()
    model = model.eval()
    actual_predictions = []
    
    for i in range(test_size-m):
        j_test_inputs = test_inputs[i:seq_len+i]
        for j in range(m):
            j_seq_pre = torch.FloatTensor(j_test_inputs[-seq_len:])
            j_seq = normiraj(j_seq_pre, j_seq_pre)
            with torch.no_grad():
                model.hidden = (torch.zeros(1,1,model.hidden_size))
                j_pred = odnormiraj(model(j_seq).item(),j_seq_pre)
                j_test_inputs.append(j_pred.item())
        actual_predictions.append((train_size + i, j_test_inputs[-m:], 0))
    return actual_predictions




def fit_slovar(slovarji):
    # Naredimo zanko ki gre čez vse mestne občine z določenimi hiperparametri, ki so podani v slovarju.
    epochs = slovarji["epochs"]
    lr = slovarji["lr"]
    hidden = slovarji["hidden"]
    seq_len = slovarji["seq_len"]
    m = slovarji["m"]
    test_size = slovarji["test_size"]
    normiranje = slovarji["normiranje"]
    metoda = slovarji["metoda"]
    actual_all_predictions = {"napovedi": {}}
    start = time.time()
    for obcina in mestne_obcine:
        actual_predicitons = fit_model(obcina = obcina, epochs = epochs, lr = lr, hidden = hidden, seq_len = seq_len, m = m, test_size = test_size, normiranje = normiranje, metoda = metoda)
        actual_all_predictions["napovedi"][obcina] = actual_predicitons
        print(obcina + ' zmodelirana')
    end = time.time()
    actual_all_predictions['time'] = end - start
    save_results(actual_predictions=actual_all_predictions, epochs = epochs, lr = lr, hidden = hidden, seq_len = seq_len, m = m, test_size = test_size, normiranje = normiranje, metoda = metoda, save_path=SAVE_PATH)

def main(sez):
    # Iteriramo čez seznam slovarjev s hiperparametri.
    for slovar in sez:
        fit_slovar(slovar)
        print(make_name(slovar) + ' zmodeliran.')

def naredi_slovarje():
    sez = []
    moj_slovar = {"metoda": "LSTM",
        "epochs": 10,
        "lr": 0.001,
        "hidden": 32,
        "seq_len": 30,
        "m": 7,
        "test_size": 200,
        "normiranje": "po_sekvencah"
    }
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(2):
                    moj_slovar = {"metoda": "LSTM",
                        "m": 7,
                        "test_size": 200,
                        "normiranje": "po_sekvencah"
                    }
                    moj_slovar["seq_len"] = 5 + j*5
                    moj_slovar["lr"] = 10**(-(2+i))
                    moj_slovar["hidden"] = 2**(4+k)
                    moj_slovar["epochs"] = 10 + l*10
                    sez.append(moj_slovar)

    moj_slovar["m"] = 30
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(2):
                    moj_slovar = {"metoda": "LSTM",
                        "m": 30,
                        "test_size": 200,
                        "normiranje": "po_sekvencah"
                    }
                    moj_slovar["seq_len"] = 10 + j*5
                    moj_slovar["lr"] = 10**(-(2+i))
                    moj_slovar["hidden"] = 2**(4+k)
                    moj_slovar["epochs"] = 10 + l*10
                    sez.append(moj_slovar)
    return sez

if __name__ == "__main__":
    start1 = time.time()
    sez_slovarjev = naredi_slovarje()
    main(sez_slovarjev)
    end1 = time.time()
    print(end1-start1)

