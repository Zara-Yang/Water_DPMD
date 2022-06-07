import numpy as np
import tables
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
# ==============  CONSTANT DEFINE  ==================
lr_init  = 0.001
pe_start = 0
pe_limit = 0
pf_start = 1
pf_limit = 1
lr_update_step = 1000
decay_rate = 0.95
decay_step = 5000

class Dataset_hdf5(Dataset):
    def __init__(self, hdf5_file_path, O_rescale_factor_path, H_rescale_factor_path, Energy = False):
        self.hdf5_file = tables.open_file(hdf5_file_path, "r")
        xO_rescale_factor = np.loadtxt(O_rescale_factor_path)
        xH_rescale_factor = np.loadtxt(H_rescale_factor_path)

        self.xO_av = xO_rescale_factor[:,0].astype(np.float32)
        self.xO_std= xO_rescale_factor[:,1].astype(np.float32)
        self.xH_av = xH_rescale_factor[:,0].astype(np.float32)
        self.xH_std= xH_rescale_factor[:,1].astype(np.float32)

        self.xO_std[self.xO_std < 0.01] = 0.01
        self.xH_std[self.xH_std < 0.01] = 0.01

        self.xO = self.hdf5_file.root.features.xO
        self.xH = self.hdf5_file.root.features.xH

        self.xOOd = self.hdf5_file.root.features.xOOd
        self.xHOd = self.hdf5_file.root.features.xHOd
        self.xOHd = self.hdf5_file.root.features.xOHd
        self.xHHd = self.hdf5_file.root.features.xHHd

        self.O_label = self.hdf5_file.root.labels.FO
        self.H_label = self.hdf5_file.root.labels.FH
        self.E_label = self.hdf5_file.root.labels.E if Energy else None

        self.nconfig = self.xO.shape[0]
    def __len__(self):
        return(self.nconfig)
    def __getitem__(self, index):
        xO_buffer = (self.xO[index] - self.xO_av) / self.xO_std
        xH_buffer = (self.xH[index] - self.xH_av) / self.xH_std

        xOOd_buffer = self.xOOd[index] / np.expand_dims(self.xO_std, axis = (0,1,2))
        xHOd_buffer = self.xHOd[index] / np.expand_dims(self.xO_std, axis = (0,1,2))
        xOHd_buffer = self.xOHd[index] / np.expand_dims(self.xH_std, axis = (0,1,2))
        xHHd_buffer = self.xHHd[index] / np.expand_dims(self.xH_std, axis = (0,1,2))

        fO_buffer = self.O_label[index]
        fH_buffer = self.H_label[index]

        E_buffer = self.E_label[index] if(self.E_label != None) else 0

        sample = {  "xO"    : xO_buffer,
                    "xH"    : xH_buffer,
                    "xOOd"  : xOOd_buffer,
                    "xHOd"  : xHOd_buffer,
                    "xOHd"  : xOHd_buffer,
                    "xHHd"  : xHHd_buffer,
                    "fO"    : fO_buffer,
                    "fH"    : fH_buffer,
                    "E"     : E_buffer}
        return(sample)

class BPNet(nn.Module):
    def __init__(self, save_path = None):
        super(BPNet, self).__init__()
        O_net = [192,100,40,10,1]
        H_net = [192,100,40,10,1]
        self.O_net = O_net
        self.H_net = H_net
        if save_path == None:
            self.Ow1 = nn.Parameter(torch.randn(O_net[0],O_net[1])/1e3)
            self.Ob1 = nn.Parameter(torch.randn(O_net[1])/1e3)
            self.Ow2 = nn.Parameter(torch.randn(O_net[1],O_net[2])/1e3)
            self.Ob2 = nn.Parameter(torch.randn(O_net[2])/1e3)
            self.Ow3 = nn.Parameter(torch.randn(O_net[2],O_net[3])/1e3)
            self.Ob3 = nn.Parameter(torch.randn(O_net[3])/1e3)
            self.Ow4 = nn.Parameter(torch.randn(O_net[3],O_net[4])/1e3)
            self.Ob4 = nn.Parameter(torch.randn(O_net[4])/1e3)

            self.Hw1 = nn.Parameter(torch.randn(H_net[0],H_net[1])/1e3)
            self.Hb1 = nn.Parameter(torch.randn(H_net[1])/1e3)
            self.Hw2 = nn.Parameter(torch.randn(H_net[1],H_net[2])/1e3)
            self.Hb2 = nn.Parameter(torch.randn(H_net[2])/1e3)
            self.Hw3 = nn.Parameter(torch.randn(H_net[2],H_net[3])/1e3)
            self.Hb3 = nn.Parameter(torch.randn(H_net[3])/1e3)
            self.Hw4 = nn.Parameter(torch.randn(H_net[3],H_net[4])/1e3)
            self.Hb4 = nn.Parameter(torch.randn(H_net[4])/1e3)
        else:
            Ow1 = np.loadtxt("{}/Ow1.txt".format(save_path))
            Ow2 = np.loadtxt("{}/Ow2.txt".format(save_path))
            Ow3 = np.loadtxt("{}/Ow3.txt".format(save_path))
            Ow4 = np.expand_dims(np.loadtxt("{}/Ow4.txt".format(save_path)), axis = 1)

            Ob1 = np.loadtxt("{}/Ob1.txt".format(save_path))
            Ob2 = np.loadtxt("{}/Ob2.txt".format(save_path))
            Ob3 = np.loadtxt("{}/Ob3.txt".format(save_path))
            Ob4 = np.expand_dims(np.loadtxt("{}/Ob4.txt".format(save_path)), axis = (0, 1))

            Hw1 = np.loadtxt("{}/Hw1.txt".format(save_path))
            Hw2 = np.loadtxt("{}/Hw2.txt".format(save_path))
            Hw3 = np.loadtxt("{}/Hw3.txt".format(save_path))
            Hw4 = np.expand_dims(np.loadtxt("{}/Hw4.txt".format(save_path)), axis = 1)

            Hb1 = np.loadtxt("{}/Hb1.txt".format(save_path))
            Hb2 = np.loadtxt("{}/Hb2.txt".format(save_path))
            Hb3 = np.loadtxt("{}/Hb3.txt".format(save_path))
            Hb4 = np.expand_dims(np.loadtxt("{}/Hb4.txt".format(save_path)), axis = (0, 1))

            self.Ow1 = nn.Parameter(torch.tensor(Ow1).to(torch.float32))
            self.Ow2 = nn.Parameter(torch.tensor(Ow2).to(torch.float32))
            self.Ow3 = nn.Parameter(torch.tensor(Ow3).to(torch.float32))
            self.Ow4 = nn.Parameter(torch.tensor(Ow4).to(torch.float32))

            self.Ob1 = nn.Parameter(torch.tensor(Ob1).to(torch.float32))
            self.Ob2 = nn.Parameter(torch.tensor(Ob2).to(torch.float32))
            self.Ob3 = nn.Parameter(torch.tensor(Ob3).to(torch.float32))
            self.Ob4 = nn.Parameter(torch.tensor(Ob4).to(torch.float32))

            self.Hw1 = nn.Parameter(torch.tensor(Hw1).to(torch.float32))
            self.Hw2 = nn.Parameter(torch.tensor(Hw2).to(torch.float32))
            self.Hw3 = nn.Parameter(torch.tensor(Hw3).to(torch.float32))
            self.Hw4 = nn.Parameter(torch.tensor(Hw4).to(torch.float32))

            self.Hb1 = nn.Parameter(torch.tensor(Hb1).to(torch.float32))
            self.Hb2 = nn.Parameter(torch.tensor(Hb2).to(torch.float32))
            self.Hb3 = nn.Parameter(torch.tensor(Hb3).to(torch.float32))
            self.Hb4 = nn.Parameter(torch.tensor(Hb4).to(torch.float32))
    def forward(self, x_O, x_H, dx_OO, dx_HO, dx_OH, dx_HH):
        x_O = x_O.unsqueeze(1).unsqueeze(2)
        x_H = x_H.unsqueeze(1).unsqueeze(2)
        z1_O = torch.matmul(x_O, self.Ow1) + self.Ob1
        z2_O = torch.matmul(torch.tanh(z1_O), self.Ow2) + self.Ob2
        z3_O = torch.matmul(torch.tanh(z2_O), self.Ow3) + self.Ob3
        z4_O = torch.matmul(torch.tanh(z3_O), self.Ow4) + self.Ob4

        z1_H = torch.matmul(x_H, self.Hw1) + self.Hb1
        z2_H = torch.matmul(torch.tanh(z1_H), self.Hw2) + self.Hb2
        z3_H = torch.matmul(torch.tanh(z2_H), self.Hw3) + self.Hb3
        z4_H = torch.matmul(torch.tanh(z3_H), self.Hw4) + self.Hb4

        ap1_OO = torch.matmul(dx_OO, self.Ow1)  / torch.cosh(z1_O) ** 2
        ap2_OO = torch.matmul(ap1_OO, self.Ow2) / torch.cosh(z2_O) ** 2
        ap3_OO = torch.matmul(ap2_OO, self.Ow3) / torch.cosh(z3_O) ** 2
        y_OO = torch.matmul(ap3_OO, self.Ow4)

        ap1_HO = torch.matmul(dx_HO, self.Ow1)  / torch.cosh(z1_O) ** 2
        ap2_HO = torch.matmul(ap1_HO, self.Ow2) / torch.cosh(z2_O) ** 2
        ap3_HO = torch.matmul(ap2_HO, self.Ow3) / torch.cosh(z3_O) ** 2
        y_HO = torch.matmul(ap3_HO, self.Ow4)

        ap1_HH = torch.matmul(dx_HH, self.Hw1)  / torch.cosh(z1_H) ** 2
        ap2_HH = torch.matmul(ap1_HH, self.Hw2) / torch.cosh(z2_H) ** 2
        ap3_HH = torch.matmul(ap2_HH, self.Hw3) / torch.cosh(z3_H) ** 2
        y_HH = torch.matmul(ap3_HH, self.Hw4)

        ap1_OH = torch.matmul(dx_OH, self.Hw1)  / torch.cosh(z1_H) ** 2
        ap2_OH = torch.matmul(ap1_OH, self.Hw2) / torch.cosh(z2_H) ** 2
        ap3_OH = torch.matmul(ap2_OH, self.Hw3) / torch.cosh(z3_H) ** 2
        y_OH = torch.matmul(ap3_OH, self.Hw4)

        y_O = torch.sum(y_OO, axis=(-1, -2)) + torch.sum(y_OH, axis=(-1, -2))
        y_H = torch.sum(y_HH, axis=(-1, -2)) + torch.sum(y_HO, axis=(-1, -2))
        eng = torch.sum(z4_O, axis = (1,2,3,4,)) + torch.sum(z4_H, axis = (1,2,3,4,))

        return y_O, y_H, eng
    def save_weight(self,save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt("{}/Ow1.txt".format(save_path), self.Ow1.detach().numpy())
        np.savetxt("{}/Ow2.txt".format(save_path), self.Ow2.detach().numpy())
        np.savetxt("{}/Ow3.txt".format(save_path), self.Ow3.detach().numpy())
        np.savetxt("{}/Ow4.txt".format(save_path), self.Ow4.detach().numpy())
        np.savetxt("{}/Hw1.txt".format(save_path), self.Hw1.detach().numpy())
        np.savetxt("{}/Hw2.txt".format(save_path), self.Hw2.detach().numpy())
        np.savetxt("{}/Hw3.txt".format(save_path), self.Hw3.detach().numpy())
        np.savetxt("{}/Hw4.txt".format(save_path), self.Hw4.detach().numpy())

        np.savetxt("{}/Ob1.txt".format(save_path), self.Ob1.detach().numpy())
        np.savetxt("{}/Ob2.txt".format(save_path), self.Ob2.detach().numpy())
        np.savetxt("{}/Ob3.txt".format(save_path), self.Ob3.detach().numpy())
        np.savetxt("{}/Ob4.txt".format(save_path), self.Ob4.detach().numpy())
        np.savetxt("{}/Hb1.txt".format(save_path), self.Hb1.detach().numpy())
        np.savetxt("{}/Hb2.txt".format(save_path), self.Hb2.detach().numpy())
        np.savetxt("{}/Hb3.txt".format(save_path), self.Hb3.detach().numpy())
        np.savetxt("{}/Hb4.txt".format(save_path), self.Hb4.detach().numpy())

def Calculate_loss_params(lr, lr_init, p_start, p_limit):
    p = (lr / lr_init) * p_start + (1 - lr / lr_init) * p_limit
    return(p)

def Exp_decay(lr_init, decay_rate, decay_step, iepoch):
    lr = lr_init * pow( decay_rate, iepoch / decay_step)
    return(lr)

def Save_config(save_path, **params):
    with open("{}/config.log".format(save_path), "w") as fp:
        fp.write("="*40 + "\n")
        fp.write("{:<20}{:<20}\n".format("Param_name", "Param_value"))
        fp.write("-"*40 + "\n")
        fp.write("{:<20}{:<20}\n".format("Date&Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        for key in params:
            fp.write("{:<20}{:<20}\n".format(key, str(params[key])))
        fp.write("="*40 + "\n")

def Train_network(train_dataset, valid_dataset, epoch_number, model_save_path, model_save_name, batch_size):
    net = BPNet()
    optimizer = optim.AdamW(net.parameters(), lr = lr_init)
    info_file = open("{}/{}.dat".format(model_save_path, model_save_name),"w")

    optimize_model = None
    optimize_loss = 1e6
    optimize_index = 0

    pe_buffer = pe_start
    pf_buffer = pf_start

    for iepoch in tqdm(range(epoch_number)):
        train_data_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_data_loader = DataLoader(valid_dataset, batch_size = len(valid_dataset), shuffle = False)
        epoch_loss_train = 0
        epoch_loss_valid = 0
        train_batch_number = 0
        valid_batch_number = 0
        for ibatch, batch_sample in enumerate(train_data_loader):
            fO_pred, fH_pred, E_pred = net(     batch_sample["xO"],
                                                batch_sample["xH"],
                                                batch_sample["xOOd"],
                                                batch_sample["xHOd"],
                                                batch_sample["xOHd"],
                                                batch_sample["xHHd"])
            loss =  pf_buffer * (torch.mean(torch.pow(fO_pred - batch_sample["fO"], 2)) + torch.mean(torch.pow(fH_pred - batch_sample["fH"],2)))# + pe_buffer * (torch.mean(torch.pow(E_pred - batch_sample["E"],2)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.detach()
            train_batch_number += 1
        for ibatch, batch_sample in enumerate(valid_data_loader):
            fO_pred, fH_pred, E_pred = net(     batch_sample["xO"],
                                                batch_sample["xH"],
                                                batch_sample["xOOd"],
                                                batch_sample["xHOd"],
                                                batch_sample["xOHd"],
                                                batch_sample["xHHd"])
            loss =  pf_buffer * (torch.mean(torch.pow(fO_pred - batch_sample["fO"], 2)) + torch.mean(torch.pow(fH_pred - batch_sample["fH"],2)))# + pe_buffer * (torch.mean(torch.pow(E_pred - batch_sample["E"],2)))
            MSEO = torch.mean(torch.pow(fO_pred - batch_sample["fO"], 2))
            MSEH = torch.mean(torch.pow(fH_pred - batch_sample["fH"], 2))
            MSEE = 0
            epoch_loss_valid += loss.detach()
            valid_batch_number += 1
        RMSEO = torch.sqrt(MSEO)
        RMSEH = torch.sqrt(MSEH)
        RMSEE = 0
        epoch_loss_train /= train_batch_number
        epoch_loss_valid /= valid_batch_number

        if epoch_loss_valid < optimize_loss:
            optimize_loss = epoch_loss_valid
            optimize_index = iepoch
            if not os.path.exists("{}/Net_params/".format(model_save_path)):
                os.makedirs("{}/Net_params/".format(model_save_path))
            net.save_weight("{}/Net_params/".format(model_save_path))

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        info_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(iepoch,epoch_loss_train,epoch_loss_valid,RMSEO,RMSEH,RMSEE,lr, pe_buffer, pf_buffer))
        print(" =================================================== ")
        print(" [ EPOCH   ] : {:>10}".format(iepoch))
        print(" [ LR      ] : {:>10}".format(lr))
        print(" [ PE &PF  ] : {:>25}{:>25}".format(pe_buffer, pf_buffer))
        print(" [ LOSS    ] : {:>25}{:>25}".format(epoch_loss_train, epoch_loss_valid))
        print(" [ RMSE FO ] : {:>25}{:>25}".format(0, RMSEO))
        print(" [ RMSE FH ] : {:>25}{:>25}".format(0, RMSEH))
        print(" [ OPIMT   ] : {:>25}{:>25}".format(optimize_index, optimize_loss))

        if iepoch % lr_update_step == 0 and iepoch != 0:
            lr = Exp_decay(init_lr, decay_rate, decay_step, iepoch)
            pe_buffer = Calculate_loss_params(lr, lr_init, pe_start, pe_limit)
            pf_buffer = Calculate_loss_params(lr, lr_init, pf_start, pf_limit)
            optimizer.param_groups[0]['lr'] = lr

if __name__ == "__main__":
    INPUT_FILE_PATH  = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/InputData"
    TRAIN_HDF5_FILE  = "{}/TrainInput.hdf5".format(INPUT_FILE_PATH)
    VALID_HDF5_FILE  = "{}/ValidInput.hdf5".format(INPUT_FILE_PATH)
    O_RESCALE_FACTOR = "{}/xO_scalefactor.txt".format(INPUT_FILE_PATH)
    H_RESCALE_FACTOR = "{}/xH_scalefactor.txt".format(INPUT_FILE_PATH)

    MODEL_SAVE_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Models/"
    MODEL_SAVE_NAME = "Model_test"

    train_data = Dataset_hdf5(TRAIN_HDF5_FILE, O_RESCALE_FACTOR, H_RESCALE_FACTOR, Energy = True)
    valid_data = Dataset_hdf5(VALID_HDF5_FILE, O_RESCALE_FACTOR, H_RESCALE_FACTOR, Energy = True)

    Train_network(  train_dataset = train_data,
                    valid_dataset = valid_data,
                    epoch_number = 10000,
                    model_save_path = MODEL_SAVE_PATH + MODEL_SAVE_NAME,
                    model_save_name = "Core_force_model",
                    batch_size = 5)

    train_data.hdf5_file.close()
