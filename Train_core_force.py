import os
import numpy as np
from tqdm import tqdm
from glob import glob
import tables
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import datetime

decay_rate = 0.95
decay_step = 5000

lr_update_frequency = 1

lr_init = 1e-3

pe_start = 1e-3
pe_limit = 1

pf_start = 100
pf_limit = 1

class Data_loader():
    def __init__(self,xO_path, xH_path, dxOO_path, dxOH_path, dxHO_path, dxHH_path, fO_path, fH_path, E_path):
        self.xO = np.load(xO_path)
        self.xH = np.load(xH_path)

        self.dxOO = np.load(dxOO_path)
        self.dxOH = np.load(dxOH_path)

        self.dxHO = np.load(dxHO_path)
        self.dxHH = np.load(dxHH_path)
        if type(fO_path) != type(None):
            self.fO = np.load(fO_path)
            self.fH = np.load(fH_path)
        else:
            self.fO = None
            self.fH = None
        if type(E_path) != type(None):
            self.E  = np.load(E_path)
        else:
            self.E = None
        self.config_number = self.xO.shape[0]
    def Generator(self,batch_size = 1):
        batch_start = 0
        batch_continue = True
        while batch_continue :
            batch_end = batch_start + batch_size
            if batch_start == self.config_number:
                break
            if batch_end > self.config_number:
                batch_end = self.config_number
                batch_continue = False
            batch_xO = torch.from_numpy(self.xO[batch_start : batch_end,:,:]).to(torch.float32)
            batch_xH = torch.from_numpy(self.xH[batch_start : batch_end,:,:]).to(torch.float32)

            batch_dxOO = torch.from_numpy(self.dxOO[batch_start : batch_end,:,:,:,:]).to(torch.float32)
            batch_dxOH = torch.from_numpy(self.dxOH[batch_start : batch_end,:,:,:,:]).to(torch.float32)

            batch_dxHO = torch.from_numpy(self.dxHO[batch_start : batch_end,:,:,:,:]).to(torch.float32)
            batch_dxHH = torch.from_numpy(self.dxHH[batch_start : batch_end,:,:,:,:]).to(torch.float32)
            if type(self.fO) != type(None):
                batch_fO = torch.from_numpy(self.fO[batch_start : batch_end,:,:]).to(torch.float32)
                batch_fH = torch.from_numpy(self.fH[batch_start : batch_end,:,:]).to(torch.float32)
            else:
                batch_fO = None
                batch_fH = None
            if type(self.E) != type(None):
                batch_E  = torch.tensor(self.E[batch_start : batch_end])
            else:
                batch_E = None

            yield(  batch_xO, batch_xH,
                    batch_dxOO, batch_dxOH,
                    batch_dxHO, batch_dxHH,
                    batch_fO, batch_fH, batch_E)
            batch_start = batch_start + batch_size

class Data_loader_hdf5():
    def __init__(self, hdf5_file_path, O_rescale_factor_path, H_rescale_factor_path):
        self.hdf5_file = table.open_file(hdf5_file_path, "r")
        self.xO_rescale_factor = np.loadtxt(O_rescale_factor_path)
        self.xH_rescale_factor = np.loadtxt(H_rescale_factor_path)

        self.xO = self.hdf5_file.root.features.xO
        self.xH = self.hdf5_file.root.features.xH


class BPNet(nn.Module):
    def __init__(self, save_path = None):
        super(BPNet, self).__init__()
        O_net = [192,100,40,10,1]
        H_net = [192,100,40,10,1]
        self.O_net = O_net
        self.H_net = H_net
        if save_path == None:

            self.Ow1 = nn.Parameter(torch.tensor(np.ones((O_net[0],O_net[1]))).to(torch.float32)/1e2)
            self.Ow2 = nn.Parameter(torch.tensor(np.ones((O_net[1],O_net[2]))).to(torch.float32)/1e2)
            self.Ow3 = nn.Parameter(torch.tensor(np.ones((O_net[2],O_net[3]))).to(torch.float32)/1e2)
            self.Ow4 = nn.Parameter(torch.tensor(np.ones((O_net[3],O_net[4]))).to(torch.float32)/1e2)
            self.Ob1 = nn.Parameter(torch.tensor(np.ones(O_net[1])).to(torch.float32)/1e2)
            self.Ob2 = nn.Parameter(torch.tensor(np.ones(O_net[2])).to(torch.float32)/1e2)
            self.Ob3 = nn.Parameter(torch.tensor(np.ones(O_net[3])).to(torch.float32)/1e2)
            self.Ob4 = nn.Parameter(torch.tensor(np.ones(O_net[4])).to(torch.float32)/1e2)

            self.Hw1 = nn.Parameter(torch.tensor(np.ones((O_net[0],O_net[1]))).to(torch.float32)/1e2)
            self.Hw2 = nn.Parameter(torch.tensor(np.ones((O_net[1],O_net[2]))).to(torch.float32)/1e2)
            self.Hw3 = nn.Parameter(torch.tensor(np.ones((O_net[2],O_net[3]))).to(torch.float32)/1e2)
            self.Hw4 = nn.Parameter(torch.tensor(np.ones((O_net[3],O_net[4]))).to(torch.float32)/1e2)
            self.Hb1 = nn.Parameter(torch.tensor(np.ones(O_net[1])).to(torch.float32)/1e2)
            self.Hb2 = nn.Parameter(torch.tensor(np.ones(O_net[2])).to(torch.float32)/1e2)
            self.Hb3 = nn.Parameter(torch.tensor(np.ones(O_net[3])).to(torch.float32)/1e2)
            self.Hb4 = nn.Parameter(torch.tensor(np.ones(O_net[4])).to(torch.float32)/1e2)

            """
            self.Ow1 = nn.Parameter(torch.randn(O_net[0],O_net[1])/1e4)
            self.Ob1 = nn.Parameter(torch.randn(O_net[1])/1e4)
            self.Ow2 = nn.Parameter(torch.randn(O_net[1],O_net[2])/1e4)
            self.Ob2 = nn.Parameter(torch.randn(O_net[2])/1e4)
            self.Ow3 = nn.Parameter(torch.randn(O_net[2],O_net[3])/1e4)
            self.Ob3 = nn.Parameter(torch.randn(O_net[3])/1e4)
            self.Ow4 = nn.Parameter(torch.randn(O_net[3],O_net[4])/1e4)
            self.Ob4 = nn.Parameter(torch.randn(O_net[4])/1e4)

            self.Hw1 = nn.Parameter(torch.randn(H_net[0],H_net[1])/1e4)
            self.Hb1 = nn.Parameter(torch.randn(H_net[1])/1e4)
            self.Hw2 = nn.Parameter(torch.randn(H_net[1],H_net[2])/1e4)
            self.Hb2 = nn.Parameter(torch.randn(H_net[2])/1e4)
            self.Hw3 = nn.Parameter(torch.randn(H_net[2],H_net[3])/1e4)
            self.Hb3 = nn.Parameter(torch.randn(H_net[3])/1e4)
            self.Hw4 = nn.Parameter(torch.randn(H_net[3],H_net[4])/1e4)
            self.Hb4 = nn.Parameter(torch.randn(H_net[4])/1e4)
            """
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

def Exp_decay(optimizer, decay_rate, decay_step, iepoch):
    lr = optimizer.param_groups[0]['lr'] * pow( decay_rate, iepoch / decay_step)
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
def Train_network(train_data, valid_data, epoch_number, model_save_path, model_save_name, batch_size):
    net = BPNet()
    optimizer = optim.AdamW(net.parameters(), lr = lr_init)
    info_file = open("{}/{}.dat".format(model_save_path, model_save_name),"w")
    # save model
    optimize_model = None
    optimize_loss = 1e6
    optimize_index = 0

    pe_buffer = pe_start
    pf_buffer = pf_start
    batch_index = 0

    for iepoch in tqdm(range(epoch_number)):
        train_generator = train_data.Generator(batch_size)
        valid_generator = valid_data.Generator(1)

        epoch_loss_train = 0
        epoch_loss_valid = 0

        MAEO = 0
        MAEH = 0
        MSEO = 0
        MSEH = 0
        MAEE = 0
        MSEE = 0
        for batch_xO, batch_xH, batch_dxOO, batch_dxOH, batch_dxHO, batch_dxHH, batch_fO, batch_fH, batch_E in train_generator:
            yO_pred, yH_pred, E_pred = net(batch_xO, batch_xH, batch_dxOO, batch_dxHO, batch_dxOH, batch_dxHH)
            if type(batch_E) != type(None):
                loss = pf_buffer * (torch.mean(torch.pow(yO_pred - batch_fO, 2)) + torch.mean(torch.pow(yH_pred - batch_fH,2))) + pe_buffer * (torch.mean(torch.pow(E_pred - batch_E,2)))
                # loss = pf_buffer * (torch.mean(torch.pow(yO_pred - batch_fO, 2)) + torch.mean(torch.pow(yH_pred - batch_fH,2)))
            else:
                loss = pf_buffer * (torch.mean(torch.pow(yO_pred - batch_fO, 2)) + torch.mean(torch.pow(yH_pred - batch_fH,2)))
            epoch_loss_train += loss.detach()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_index += 1
        for batch_xO, batch_xH, batch_dxOO, batch_dxOH, batch_dxHO, batch_dxHH, batch_fO, batch_fH, batch_E in valid_generator:
            yO_pred, yH_pred, E_pred = net(batch_xO, batch_xH, batch_dxOO, batch_dxHO, batch_dxOH, batch_dxHH)
            MSEO += torch.mean(torch.pow(yO_pred - batch_fO, 2))
            MSEH += torch.mean(torch.pow(yH_pred - batch_fH, 2))
            MAEO += torch.mean(torch.abs(yO_pred - batch_fO))
            MAEH += torch.mean(torch.abs(yH_pred - batch_fH))
            if type(batch_E) != type(None):
                loss = pf_buffer * (torch.mean(torch.pow(yO_pred - batch_fO, 2)) + torch.mean(torch.pow(yH_pred - batch_fH,2))) + pe_buffer * (torch.mean(torch.pow(E_pred - batch_E,2)))
                # loss = pf_buffer * (torch.mean(torch.pow(yO_pred - batch_fO, 2)) + torch.mean(torch.pow(yH_pred - batch_fH,2)))
                MSEE += torch.mean(torch.pow( E_pred - batch_E,  2))
                MAEE += torch.mean(torch.abs( E_pred - batch_E ))
            else:
                loss = pf_buffer * (torch.mean(torch.pow(yO_pred - batch_fO, 2)) + torch.mean(torch.pow(yH_pred - batch_fH,2)))
                print("Force only")
            epoch_loss_valid += loss.detach()
        MSEO = MSEO / valid_data.config_number
        MSEH = MSEH / valid_data.config_number
        MSEE = MSEE / valid_data.config_number

        MAEO = MAEO / valid_data.config_number
        MAEH = MAEH / valid_data.config_number
        MAEE = MAEE / valid_data.config_number

        RMSEO = torch.sqrt(MSEO)
        RMSEH = torch.sqrt(MSEH)
        RMSEE = torch.sqrt(torch.tensor(MSEE).detach())

        epoch_loss_valid = epoch_loss_valid / ( valid_data.config_number)
        epoch_loss_train = epoch_loss_train / ( train_data.config_number / batch_size)

        if epoch_loss_valid < optimize_loss:
            optimize_loss = epoch_loss_valid
            optimize_index = iepoch
            torch.save(net.state_dict(), "{}/{}.pth".format(model_save_path,model_save_name))
            net_traced = torch.jit.trace(net, (batch_xO, batch_xH, batch_dxOO, batch_dxHO, batch_dxOH, batch_dxHH))
            net_traced.save("{}/{}.pt".format(model_save_path,model_save_name))
            if not os.path.exists("{}/Core_weights/".format(model_save_path)):
                os.makedirs("{}/Core_weights/".format(model_save_path))
            net.save_weight("{}/Core_weights/".format(model_save_path))

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        info_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(iepoch,epoch_loss_train,epoch_loss_valid,RMSEO,RMSEH,RMSEE,MAEO,MAEH,MAEE,lr, pe_buffer, pf_buffer))
        info_file.flush()
        print(" ========================================================================== ")
        print(" [ EPOCH   ] : {:>10}".format(iepoch))
        print(" [ LR      ] : {:>10}".format(lr))
        print(" [ PE &PF  ] : {:>25}{:>25}".format(pe_buffer, pf_buffer))
        print(" [ LOSS    ] : {:>25}{:>25}".format(epoch_loss_train, epoch_loss_valid))
        print(" [ RMSE FO ] : {:>25}{:>25}".format(0, RMSEO))
        print(" [ RMSE FH ] : {:>25}{:>25}".format(0, RMSEH))
        print(" [ RMSE  E ] : {:>25}{:>25}".format(0, RMSEE))
        print(" [ MAE  FO ] : {:>25}{:>25}".format(0, MAEO))
        print(" [ MAE  FH ] : {:>25}{:>25}".format(0, MAEH))
        print(" [ MAE   E ] : {:>25}{:>25}".format(0, MAEE))
        print(" [ OPIMT   ] : {:>25}{:>25}".format(optimize_index, optimize_loss))
        if iepoch % lr_update_frequency == 0 and iepoch != 0:
            lr = Exp_decay(optimizer, decay_rate, decay_step, iepoch)
            pe_buffer = Calculate_loss_params(lr, lr_init, pe_start, pe_limit)
            pf_buffer = Calculate_loss_params(lr, lr_init, pf_start, pf_limit)
            optimizer.param_groups[0]['lr'] = lr
if __name__ == "__main__":
    TRAIN_DATA_FOLDER = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/TrainInput"
    VALID_DATA_FOLDER = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/ValidInput"
    MODEL_SAVE_FOLDER = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Models/Model_test"
    if not os.path.exists(MODEL_SAVE_FOLDER):
        os.makedirs(MODEL_SAVE_FOLDER)
    batch_size = 5
    total_epoch = 10000

    Save_config(save_path  = MODEL_SAVE_FOLDER,
                train_data = TRAIN_DATA_FOLDER,
                valid_data = VALID_DATA_FOLDER,
                lr = lr_init,
                decay_rate = decay_rate,
                decay_step = decay_step,
                update_step = lr_update_frequency,
                pe_start = pe_start,
                pe_limit = pe_limit,
                pf_start = pf_start,
                pf_limit = pf_limit,
                network_O = BPNet().O_net,
                network_H = BPNet().H_net,
                batch_size = batch_size,
                total_epoch = total_epoch,
                descriptor = "Train total force only(eV/A), ignore energy, use DPMD example sumed up data")
    train_dataset = Data_loader(xO_path="{}/xO.npy".format(TRAIN_DATA_FOLDER),
                                xH_path="{}/xH.npy".format(TRAIN_DATA_FOLDER),
                                dxOO_path="{}/xOO.npy".format(TRAIN_DATA_FOLDER),
                                dxOH_path="{}/xOH.npy".format(TRAIN_DATA_FOLDER),
                                dxHO_path="{}/xHO.npy".format(TRAIN_DATA_FOLDER),
                                dxHH_path="{}/xHH.npy".format(TRAIN_DATA_FOLDER),
                                fO_path = "{}/FO_total.npy".format(TRAIN_DATA_FOLDER),
                                fH_path = "{}/FH_total.npy".format(TRAIN_DATA_FOLDER),
                                E_path  = "{}/Eng_total.npy".format(TRAIN_DATA_FOLDER))
    valid_dataset = Data_loader(xO_path="{}/xO.npy".format(VALID_DATA_FOLDER),
                                xH_path="{}/xH.npy".format(VALID_DATA_FOLDER),
                                dxOO_path="{}/xOO.npy".format(VALID_DATA_FOLDER),
                                dxOH_path="{}/xOH.npy".format(VALID_DATA_FOLDER),
                                dxHO_path="{}/xHO.npy".format(VALID_DATA_FOLDER),
                                dxHH_path="{}/xHH.npy".format(VALID_DATA_FOLDER),
                                fO_path = "{}/FO_total.npy".format(VALID_DATA_FOLDER),
                                fH_path = "{}/FH_total.npy".format(VALID_DATA_FOLDER),
                                E_path  = "{}/Eng_total.npy".format(VALID_DATA_FOLDER))
    print("Data load finish")
    Train_network(  train_dataset,
                    valid_dataset,
                    total_epoch,
                    MODEL_SAVE_FOLDER,
                    "Water_core_force_model",
                    batch_size)







