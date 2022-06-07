import os
import numpy as np
from tqdm import tqdm
from glob import glob
import subprocess
import random
import math

nAtom = 192
nO = 64
nH = 128
nW = 256

O_threshold = 16
H_threshold = 32

def Produce_split_features(data_path,script_path):
    """
    Copy and execute Produce_features.o in every config folder

    Input :
    Name                    description                               dimension
    --------------------------------------------------------------------------------
    data_path       folder path which conrtain config folder            None
    script_path     cpp executable file path                            None

    Output:
        None
    """
    config_path_list = list(i for i in glob("{}/*".format(data_path)))
    for config_path in tqdm(config_path_list):
        print(config_path)
        subprocess.call(["cp",script_path,config_path])
        subprocess.call(["./{}".format(os.path.basename(script_path))],cwd=config_path)

class Data_assemble():
    @staticmethod
    def Assemble_single_feature(config_folder_path,config_list,save_folder_path,feature_name, center_type, threshold_num = 16):
        if center_type == "O" :
            ncenter = nO
        elif center_type == "H":
            ncenter = nH
        features_all = ()
        dfeatures_all = ()
        for config_folder_name in tqdm(config_list):
            feature_path = "{}/{}/features/feature_{}.txt".format(config_folder_path,config_folder_name,feature_name)
            feature_d_path = "{}/{}/features/feature_d{}.txt".format(config_folder_path,config_folder_name,feature_name)
            features  = np.loadtxt(feature_path, dtype=np.float32)
            dfeatures = np.loadtxt(feature_d_path, dtype=np.float32)
            dfeatures = dfeatures.reshape((ncenter, (O_threshold + H_threshold) * 4, nAtom, 3))
            if len(features.shape) == 1:
                features = np.expand_dims(features,axis=0)
            features_all += (features, )
            dfeatures_all += (dfeatures, )
        features_all = np.stack(features_all, axis=0)   # now it is (nconfig, ncenter, nfeatures)
        dfeatures_all = np.transpose(np.stack(dfeatures_all, axis=0), axes=(0, 1, 3, 4, 2))  # now it is (nconfig, ncenter, natoms, 3, nfeatures)
        return(features_all, dfeatures_all)
    @staticmethod
    def Assemble_all_features(features_name_tuple, features_folder_path):
        features_all = ()
        features_d_all = ()
        for feature_name in tqdm(features_name_tuple):
            features_d = np.load("{}/features_d{}".format(features_folder_path,feature_name) + ".npy")
            features = np.load("{}/features_{}".format(features_folder_path,feature_name) + ".npy")
            features_all += (features,)
            features_d_all += (features_d,)
        features_all = np.concatenate(features_all, axis=-1)  # stack along the nfeatures axis
        features_d_all = np.transpose(np.concatenate(features_d_all, axis=-1), axes=(0, 2, 3, 1, 4))
        # stack along the nfeatures axis, then make sure the number of center atoms is at the second last axis
        return features_all, features_d_all
    @staticmethod
    def Collect_features(config_folder_path, config_list, save_folder_path, features_list):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        for feature_name in features_list:
            features_all, dfeatures_all = Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name)
            np.save("{}/features_{}".format(feature_path,feature_name), features_all)
            np.save("{}/features_d{}".format(feature_path,feature_name), dfeatures_all)
    @staticmethod
    def dfeature_split(xO_d, xH_d):
        xOO_d = xO_d[:,:nO,:,:,:]
        xHO_d = xO_d[:,nO:nAtom,:,:,:]
        xOH_d = xH_d[:,:nO,:,:,:]
        xHH_d = xH_d[:,nO:nAtom,:,:,:]
        return(xOO_d, xHO_d, xOH_d, xHH_d)
    @staticmethod
    def Generate_dataset(config_folder_path, config_list, features_list, save_folder_path, reassemble = False, save_rescale_factor = False):
        if reassemble:
            Data_assemble.Collect_features(config_folder_path, config_list, save_folder_path, features_list)

        feature_path = "{}/features".format(save_folder_path)
        xO,xO_d = Data_assemble.Assemble_all_features(["O"], feature_path)
        xH,xH_d = Data_assemble.Assemble_all_features(["H"], feature_path)

        xOO_d, xHO_d, xOH_d, xHH_d = Data_assemble.dfeature_split(xO_d, xH_d)

        xO_av  = np.mean(xO, axis = (0, 1))
        xO_std = np.std(xO, axis = (0, 1))
        xH_av  = np.mean(xH, axis = (0, 1))
        xH_std = np.std(xH, axis = (0, 1))

        if save_rescale_factor :
            np.savetxt("{}/xO_scalefactor.txt".format(save_folder_path), np.stack((xO_av, xO_std), axis=-1))
            np.savetxt("{}/xH_scalefactor.txt".format(save_folder_path), np.stack((xH_av, xH_std), axis=-1))

        np.save("{}/xO".format(save_folder_path), xO)
        np.save("{}/xOO".format(save_folder_path), xOO_d)
        np.save("{}/xHO".format(save_folder_path), xHO_d)
        np.save("{}/xH".format(save_folder_path), xH)
        np.save("{}/xOH".format(save_folder_path), xOH_d)
        np.save("{}/xHH".format(save_folder_path), xHH_d)
        return(xO_av, xH_av, xO_std, xH_std)
    @staticmethod
    def Generate_Split_dataset(config_folder_path, config_list, features_list, save_folder_path, split_number = 10, reassemble = True):
        random.shuffle(config_list)
        bin_buffer = math.ceil( len(config_list) / split_number )
        data_index = {}

        xO_av_list  = ()
        xH_av_list  = ()
        xO_std_list = ()
        xH_std_list = ()
        for idata in range(split_number):
            data_index[idata] = [idata * bin_buffer, (idata + 1) * bin_buffer]
        for idata in data_index:
            start_index, end_index = data_index[idata]
            save_path = "{}/Split_dataset/{}".format(save_folder_path, idata)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            xO_av, xH_av, xO_std, xH_std = Data_assemble.Generate_dataset(config_folder_path,
                                                            config_list[start_index: end_index],
                                                            features_list,
                                                            save_path,
                                                            reassemble,
                                                            save_rescale_factor = False)
            xO_av_list  += (xO_av,)
            xH_av_list  += (xH_av,)
            xO_std_list += (xO_std,)
            xH_std_list += (xH_std,)
        xO_av_list = np.array(xO_av_list)
        xH_av_list = np.array(xH_av_list)
        xO_std_list = np.array(xO_std_list)
        xH_std_list = np.array(xH_std_list)

        xO_av = xO_av_list.mean(axis = 0)
        xH_av = xH_av_list.mean(axis = 0)
        xO_std = xO_std_list.mean(axis = 0)
        xO_std = xO_std_list.mean(axis = 0)

        np.savetxt("{}/xO_scalefactor.txt".format(save_folder_path), np.stack((xO_av, xO_std), axis=-1))
        np.savetxt("{}/xH_scalefactor.txt".format(save_folder_path), np.stack((xH_av, xH_std), axis=-1))


if __name__ == "__main__":
    config_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Split_data"
    script_name = "Calculate_features.o"
    train_input_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/TrainInput_split"
    valid_input_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/ValidInput"

    Train_config_list = [i for i in range(20)]
    Valid_config_list = [i for i in range(350,400)]

    #Produce_split_features(config_path, script_name)
    # exit()
    Data_assemble.Generate_Split_dataset(config_folder_path = config_path,
                                         config_list = Train_config_list,
                                         features_list = ["O", "H"],
                                         save_folder_path = train_input_path,
                                         split_number = 2,
                                         reassemble = True)
    exit()
    """
    Data_assemble.Generate_dataset( config_folder_path = config_path,
                                    config_list = Train_config_list,
                                    features_list = ["O", "H"],
                                    save_folder_path = train_input_path,
                                    reassemble = True,
                                    save_rescale_factor = True)
    Data_assemble.Generate_dataset( config_folder_path = config_path,
                                    config_list = Valid_config_list,
                                    features_list = ["O", "H"],
                                    save_folder_path = valid_input_path,
                                    reassemble = True,
                                    save_rescale_factor = False)
    """

