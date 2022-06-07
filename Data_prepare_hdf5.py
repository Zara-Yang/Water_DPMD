import os
import tables
import random
import numpy as np
from tqdm import tqdm
from glob import glob

nconfig = 5000
nAtom = 192
nO    = 64
nH    = 128

nfeature = 192

def Data_generator(config_folder_path, config_list, feature_name, center_type):
    if center_type == "O":
        ncenter = nO
    if center_type == "H":
        ncenter = nH
    bar = tqdm(total = len(config_list))
    for iconfig, config_name in enumerate(config_list):
        config_path = "{}/{}".format(config_folder_path, config_name)
        feature_path = "{}/features/feature_{}.txt".format(config_path, feature_name)
        dfeature_path = "{}/features/feature_d{}.txt".format(config_path, feature_name)
        force_total = "{}/{}force.txt".format(config_path, feature_name)

        features = np.loadtxt(feature_path, dtype = np.float32)
        dfeatures = np.loadtxt(dfeature_path, dtype = np.float32)
        dfeatures = dfeatures.reshape((ncenter, nfeature, nAtom, 3))
        dfeatures = np.transpose(dfeatures, axes = (2, 3, 0, 1))
        force_t = np.loadtxt(force_total)
        yield(iconfig, config_name, features, dfeatures, force_t)
        bar.update(1)

def Energy_generator(config_folder_path, config_list):
    bar = tqdm(total = len(config_list))
    for iconfig, config_name in enumerate(config_list):
        config_path = "{}/{}".format(config_folder_path, config_name)
        Energy_path = "{}/Eng.txt".format(config_path)
        energy = np.loadtxt(Energy_path)
        yield(iconfig, config_name, np.expand_dims(energy, axis = 0))
        bar.update(1)

def Cook_dataset(save_folder_path, title, generator_O, generator_H, generator_E = None, rescale_factor = True):
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    filters=tables.Filters(complevel=5,complib='blosc')
    hdf5_file       = tables.open_file("{}/{}.hdf5".format(save_folder_path,title), "w", title)
    group_features  = hdf5_file.create_group("/","features")
    group_labels    = hdf5_file.create_group("/","labels")
    if generator_E != None:
        for iconfig, config_name, energy in generator_E:
            if iconfig == 0:
                Eng = hdf5_file.create_earray(group_labels, "E", tables.Float32Atom(), shape = (0,))
            Eng.append(energy)
    for iconfig, config_name, feature, dfeature, label in generator_O:
        if iconfig == 0 :
            xO   = hdf5_file.create_earray(group_features, "xO",tables.Atom.from_dtype(feature.dtype),shape = (0, nO, nfeature))
            xOOd = hdf5_file.create_earray(group_features, "xOOd",tables.Atom.from_dtype(dfeature.dtype),shape = (0, nO ,3 , nO, nfeature))
            xHOd = hdf5_file.create_earray(group_features, "xHOd",tables.Atom.from_dtype(dfeature.dtype),shape = (0, nH ,3 , nO, nfeature))
            FO   = hdf5_file.create_earray(group_labels, "FO",tables.Atom.from_dtype(label.dtype),shape = (0, nO ,3))
        FO.append(np.expand_dims(label, axis = 0))
        xO.append(np.expand_dims(feature, axis = 0))
        xOOd.append(np.expand_dims(dfeature[:nO, :, :, :], axis = 0))
        xHOd.append(np.expand_dims(dfeature[nO:nAtom, :, :, :], axis = 0))
    for iconfig, config_name, feature, dfeature, label in generator_H:
        if iconfig == 0 :
            xH   = hdf5_file.create_earray(group_features, "xH",tables.Atom.from_dtype(feature.dtype),shape = (0, nH, nfeature))
            xOHd = hdf5_file.create_earray(group_features, "xOHd",tables.Atom.from_dtype(dfeature.dtype),shape = (0, nO ,3 , nH, nfeature))
            xHHd = hdf5_file.create_earray(group_features, "xHHd",tables.Atom.from_dtype(dfeature.dtype),shape = (0, nH ,3 , nH, nfeature))
            FH   = hdf5_file.create_earray(group_labels, "FH",tables.Atom.from_dtype(label.dtype),shape = (0, nH ,3))
        FH.append(np.expand_dims(label, axis = 0))
        xH.append(np.expand_dims(feature, axis = 0))
        xOHd.append(np.expand_dims(dfeature[:nO, :, :, :], axis = 0))
        xHHd.append(np.expand_dims(dfeature[nO:nAtom, :, :, :], axis = 0))
    if rescale_factor:
        xO_av = np.mean(hdf5_file.root.features.xO[:], axis = (0, 1))
        xO_std = np.std(hdf5_file.root.features.xO[:], axis = (0, 1))
        xH_av = np.mean(hdf5_file.root.features.xH[:], axis = (0, 1))
        xH_std = np.std(hdf5_file.root.features.xH[:], axis = (0, 1))

        np.savetxt("{}/xO_scalefactor.txt".format(save_folder_path), np.stack((xO_av, xO_std), axis=-1))
        np.savetxt("{}/xH_scalefactor.txt".format(save_folder_path), np.stack((xH_av, xH_std), axis=-1))
    hdf5_file.flush()
    hdf5_file.close()

if __name__ == "__main__":
    CONFIG_FOLDER_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data_Yihao/Split_data/"
    DATA_SAVE_PATH     = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data_Yihao/InputData/"

    total_config = [int(i) for i in range(3300)]
    train_config_list = total_config[:3000]
    valid_config_list = total_config[3000:]

    O_generator_train = Data_generator( config_folder_path = CONFIG_FOLDER_PATH,
                                        config_list = train_config_list,
                                        feature_name = "O",
                                        center_type = "O")
    H_generator_train = Data_generator( config_folder_path = CONFIG_FOLDER_PATH,
                                        config_list = train_config_list,
                                        feature_name = "H",
                                        center_type = "H")
    """
    E_generator_train = Energy_generator(   config_folder_path = CONFIG_FOLDER_PATH,
                                            config_list = train_config_list)
    """
    O_generator_valid = Data_generator( config_folder_path = CONFIG_FOLDER_PATH,
                                        config_list = valid_config_list,
                                        feature_name = "O",
                                        center_type = "O")
    H_generator_valid = Data_generator( config_folder_path = CONFIG_FOLDER_PATH,
                                        config_list = valid_config_list,
                                        feature_name = "H",
                                        center_type = "H")
    """
    E_generator_valid = Energy_generator(   config_folder_path = CONFIG_FOLDER_PATH,
                                            config_list = valid_config_list)
    """
    Cook_dataset(   save_folder_path = DATA_SAVE_PATH,
                    title = "TrainInput",
                    generator_O = O_generator_train,
                    generator_H = H_generator_train,
                    rescale_factor = True)
    Cook_dataset(   save_folder_path = DATA_SAVE_PATH,
                    title = "ValidInput",
                    generator_O = O_generator_valid,
                    generator_H = H_generator_valid,
                    rescale_factor = False)
