import os
import numpy as np
from glob import glob
from tqdm import tqdm

class Generate_label():
    @staticmethod
    def Load_single_force(config_path):
        fO_total = np.loadtxt("{}/Oforce.txt".format(config_path))
        fH_total = np.loadtxt("{}/Hforce.txt".format(config_path))

        # fO_total = fO_total * 496137.2263112725 # transfer a.u. to 10 J/(Ans * mol)
        # fH_total = fH_total * 496137.2263112725 # transfer a.u. to 10 J/(Ans * mol)

        # fO_long = np.loadtxt("{}/Oforce_l.txt".format(config_path))
        # fH_long = np.loadtxt("{}/Hforce_l.txt".format(config_path))

        # fO_short = fO_total - fO_long
        # fH_short = fH_total - fH_long

        fO_short = None
        fH_short = None
        return(fO_total, fH_total, fO_short, fH_short)
    @staticmethod
    def Assemble_force(folder_path, config_list, save_path):
        Oforce_total = ()
        Hforce_total = ()
        Oforce_short = ()
        Hforce_short = ()

        for config_name in config_list:
            config_path = "{}/{}".format(folder_path, config_name)
            fO_total, fH_total, fO_short, fH_short= Generate_label.Load_single_force(config_path)

            Oforce_total += (fO_total,)
            Hforce_total += (fH_total,)
            Oforce_short += (fO_short,)
            Hforce_short += (fH_short,)

        Oforce_total = np.array(Oforce_total)
        Hforce_total = np.array(Hforce_total)
        Oforce_short = np.array(Oforce_short)
        Hforce_short = np.array(Hforce_short)

        np.save("{}/FO_total".format(save_path), Oforce_total)
        np.save("{}/FH_total".format(save_path), Hforce_total)
        # np.save("{}/FO_short".format(save_path), Oforce_short)
        # np.save("{}/FH_short".format(save_path), Hforce_short)
    @staticmethod
    def Load_Energy(config_path):
        eng = np.loadtxt("{}/Eng.txt".format(config_path))
        return(eng)
    @staticmethod
    def Assemble_energy(folder_path, config_list, save_path):
        energy_total = ()
        for config_name in config_list:
            config_path = "{}/{}".format(folder_path, config_name)
            eng= Generate_label.Load_Energy(config_path)
            energy_total += (eng, )
        energy_total = np.array(energy_total)
        np.save("{}/Eng_total".format(save_path), energy_total)

if __name__ == "__main__":
    TRAIN_INPUT_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/TrainInput_normal"
    VALID_INPUT_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/ValidInput_normal"

    train_config_name = [i for i in range(0 , 350)]
    valid_config_name = [i for i in range(350, 400)]

    Generate_label.Assemble_force(  "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Split_data",
                                    train_config_name,
                                    TRAIN_INPUT_PATH)

    Generate_label.Assemble_force(  "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Split_data",
                                    valid_config_name,
                                    VALID_INPUT_PATH)
    exit()
    Generate_label.Assemble_energy( "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Split_data",
                                    train_config_name,
                                    TRAIN_INPUT_PATH)
    Generate_label.Assemble_energy( "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_DPMD/data/Split_data",
                                    valid_config_name,
                                    VALID_INPUT_PATH)





