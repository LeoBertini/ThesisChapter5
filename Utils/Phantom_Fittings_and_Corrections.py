#  This software was developed by Leonardo Bertini (l.bertini@nhm.ac.uk) at the Natural History Museum (London,UK).
#
#  This is released as Supporting Material as part of the following publication:
#  "XXXXXX" (link to paper and DOI once published).
#  #
#
#  Copyright (c) 2023.
#
#  The code is distributed under the MIT license https://en.wikipedia.org/wiki/MIT_License

import numpy as np
import time
import os
import pandas as pd
import warnings
import multiprocessing
from pynverse import inversefunc
from scipy.optimize import curve_fit
from tkinter import filedialog
from tkinter import *
from pathlib import Path, PureWindowsPath
from sys import platform
import matplotlib.pyplot as plt


def get_scan_name(folder_name, dir_standard_names):
    # dir_standard_names = ['CWI_Cores', 'CWI_Coral_Cores', 'NHM_fossils', 'NHM_scans']
    for dir_tag in dir_standard_names:
        if dir_tag in folder_name:
            # print(dir_tag)
            scan_name = str(PureWindowsPath(folder_name)).split(str(PureWindowsPath(dir_tag)))[-1].split('\\')[
                1]  # works on both Windows and Unix
            # scan_name = folder_name.split(dir_tag)[-1].split('\\')[1]
    return scan_name


def find_folders_by_filetype(target_file_type):
    # Selecting parent folder where scan folders are
    root = Tk()
    root.withdraw()
    main_dir = filedialog.askdirectory(title='Select the Parent folder where Scans are saved')
    root.update()

    folder_list = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.abspath(main_dir)):
        for directory in dirnames:
            for file in os.listdir(os.path.join(dirpath, directory)):
                if file.endswith(
                        target_file_type):  # pass everything to lower case in case dirnames are either upper or lower case
                    # print(os.path.join(dirpath, file))
                    target_dir = os.path.join(dirpath, directory, file)
                    folder_list.append(os.path.dirname(target_dir))
                    print(f"Found Scan folder {os.path.dirname(target_dir)}")

    folder_list = np.unique(folder_list).tolist()

    # select Spreadsheet with calibration coefficients
    root = Tk()
    root.withdraw()
    calib_dir = filedialog.askopenfilename(title='Select the spreadsheet with calibration coefficients')
    root.update()
    print(f"Calibration file indicated as {calib_dir}")

    return folder_list, os.path.abspath(main_dir), os.path.abspath(calib_dir)


def get_vsize_from_CT_filetypes(folder):
    file_extensions = [".xtekct", ".xtekVolume"]
    TargetStrings = ['VoxelSizeX=', 'Voxel size = ']
    # parent_folder = os.path.dirname(folder)

    # MAIN_PATH=os.path.join(Drive_Letter, main_dir)

    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if any([name.endswith(extension) for extension in file_extensions]):
                print(f"Found config file for scan in {os.path.abspath(os.path.join(root, name))}")
                target_file_path = os.path.abspath(os.path.join(root, name))
                ##TODO get voxelsize from xtect or CWI files or xteck volume files

    dummy_size = []
    with open(target_file_path, 'rt') as myfile:  # Open lorem.txt for reading text
        contents = myfile.read()  # Read the entire file to a string
        for each_line in contents.split("\n"):
            if any([item in each_line for item in TargetStrings]):
                # print(each_line)
                dummy_size = each_line
                break

    if TargetStrings[0] in dummy_size:
        voxelsize = float(dummy_size.split(TargetStrings[0])[-1])
    if TargetStrings[1] in dummy_size:
        voxelsize = float(dummy_size.split(TargetStrings[1])[-1])
    print(f"Voxel size is {voxelsize}")

    return voxelsize


def func_poly3(x, a, b, c, d):
    return a * (x ** 3) + b * (x ** 2) + c * x + d


def save_weights_extended_case(scan_folder, calib_dir, project_dir_list):
    warnings.filterwarnings("ignore")

    Calib_data = pd.read_excel(calib_dir)
    print('Assuming an extended phantom for file and fitting density calibrations based on 3-degree polynomial fits')
    print(scan_folder)

    Fit_dic = {'Scan_name': [],
               'Calibration_File_From': [],
               'Density_vals': [],
               'Grey_vals': [],
               'Density_corrected': [],
               'Weight_corrected': [],
               'Volume_estimate': [],
               'Histogram_Source_File': []}

    ResultsDF = pd.DataFrame(Fit_dic)
    ResultsDF['Density_vals'] = ResultsDF['Density_vals'].astype(object)
    ResultsDF['Grey_vals'] = ResultsDF['Grey_vals'].astype(object)
    ResultsDF['Density_corrected'] = ResultsDF['Density_corrected'].astype(object)
    ResultsDF['Weight_corrected'] = ResultsDF['Weight_corrected'].astype(object)
    ResultsDF['Histogram_Source_File'] = ResultsDF['Histogram_Source_File'].astype(object)

    # BUILDING DICTIONARY WITH RESULTS TO FACILITATE PLOTTING

    scan_name = get_scan_name(folder_name=scan_folder,
                              dir_standard_names=project_dir_list)

    # CALCULATE VIRTUAL WEIGHT FROM Histogram applying different functions
    # for each histogram in scanfolder

    # Find and read each histogram csv data for selected scan, then calculate virtual weight
    csv_files = []

    path_for_csvs = os.path.abspath(os.path.join(scan_folder.split(scan_name)[0], scan_name))
    for file in os.listdir(path_for_csvs):
        if 'Histogram' in file:
            # print(file)
            csv_files.append(os.path.join(path_for_csvs, file))  # list of csv histogram datasets

    if csv_files:  # not if empty list
        result_row = 1

        features_names = []
        features_avg_density = []
        features_avg_grey = []

        for csv_file in csv_files:  # each csv inside scanfolder
            scan_name_patched = csv_file.split('Histogram-')[-1].split('.csv')[0]
            print('Histograms found... importing data from csv files...')
            print(csv_file)
            print(f"Calculating coral weight based on {Calib_data['FitType'][0]}")
            Histogram_key_scan = 'Y'

            # slice histogram dataframe to get the linear count and not log scale data (export from avizo gives both)
            hist_scan = pd.read_csv(csv_file, header=None)
            for index in range(len(hist_scan[0])):
                if hist_scan[0][index] == 65535:
                    # print(index)
                    hist_scan_filtered = hist_scan[:index + 1]
                    break

            ##Getting the weight per bin
            # reference scan
            col_names = hist_scan_filtered.columns

            voxel_size = get_vsize_from_CT_filetypes(path_for_csvs)
            voxel_volume = (voxel_size ** 3) / 1000  # in cm3

            for item in range(0, len(Calib_data)):  # for each fit type in the Dataframe
                # go through calibration spreadsheet and extract coefficients for the scan being analysed
                if Calib_data['Scan_name'][item] in scan_name_patched:
                    coefficients = pd.eval(Calib_data['Coefficients_High_Low_Order'][item])
                    corr_factor = Calib_data['Density_Correction_Factor'][item]

                    a, b, c, d = coefficients
                    print('Calibration coefficients found')
                    print(coefficients)
                    break

            func_p = (lambda x, a, b, c, d: a * (x ** 3) + b * (
                        x ** 2) + c * x + d)  # define function to find inverse with the coefficients found

            # fitting inverse curve to retreive grays
            den_inserts = [0.1261, 0.26, 0.904, 1.13, 1.26, 1.44, 1.65, 1.77, 1.92, 2.7]
            Grey_Inserts = [func_p(item, a, b, c, d) for item in den_inserts]
            a1, b1, c1, d1 = "", "", "", ""
            popt, pcov = curve_fit(func_poly3, Grey_Inserts, den_inserts, maxfev=5000)
            a1, b1, c1, d1 = popt[0], popt[1], popt[2], popt[3]  # coefficients for inverse (Grey-->Density)

            # find inverse to get density estimate from gray value in domain of curve
            inverse_func = inversefunc(func_p, args=(a, b, c, d))

            print(
                '\nCalculating virtual weight for the following objects histogram and applying density correction:')
            print(csv_file)
            weight_corrected = []
            weight_uncorrected = []
            vol = []
            density_pred = []
            density_bin_corrected = []
            weight_bin_corrected = []

            for line in range(1, len(hist_scan_filtered[col_names[0]])):
                grey = int(np.floor(hist_scan_filtered[col_names[0]][line]))
                count = hist_scan_filtered[col_names[1]][line]

                # find density prediction from grey (essentially the inverted fit)
                # density_estimate = abs(inverse_func(grey))  # Inverse_Function contains a list of inverted functions in memory
                density_estimate = func_p(grey, a1, b1, c1, d1)
                if grey < Grey_Inserts[0]:
                    density_estimate = 0
                # todo amend for non-real domain (inflexions)

                density_pred.append(float(density_estimate))
                density_bin_corrected.append(density_estimate * corr_factor)
                # They are arranged in same order as the ones in the dictionary
                # no need to feed coefficients
                weight_uncorrected.append(density_estimate * abs(count) * voxel_volume)
                weight_bin_corrected.append(density_estimate * corr_factor * abs(count) * voxel_volume)
                weight_corrected.append(weight_bin_corrected)

                vol_bin = abs(count) * voxel_volume
                vol.append(vol_bin)

            print('Total estimated weight')
            print(sum(weight_bin_corrected))
            print('Total estimated volume')
            print(sum(vol))
            print('\n')

            x_grey = list(range(1, 2 ** 16 - 1))
            features_names.append(os.path.basename(csv_file).split('Histogram-')[1].split('.csv')[0])
            features_avg_density.append(sum(weight_bin_corrected) / sum(vol))
            features_avg_grey.append(np.interp((sum(weight_bin_corrected) / sum(vol)), density_pred, x_grey))

            ResultsDF.at[result_row, 'Scan_name'] = scan_name_patched
            ResultsDF.at[result_row, 'Weight_estimate'] = (sum(weight_uncorrected))
            ResultsDF.at[result_row, 'Volume_estimate'] = (sum(vol))
            ResultsDF.at[result_row, 'Calibration_File_From'] = Calib_data['FitType'][item]
            ResultsDF.at[result_row, 'Grey_vals'] = list(range(1, 2 ** 16))
            ResultsDF.at[result_row, 'Density_vals'] = density_pred
            ResultsDF.at[result_row, 'Density_corrected'] = density_bin_corrected
            ResultsDF.at[result_row, 'Weight_corrected'] = sum(weight_bin_corrected)
            ResultsDF.at[result_row, 'Histogram_Source_File'] = os.path.basename(csv_file)

            result_row += 1

        # saving results
        ResultsDF.to_excel(os.path.join(path_for_csvs,
                                        'Results_Density_Corrected_' + scan_name_patched + '_BasedOn_' +
                                        Calib_data['FitType'][item] + '.xlsx'), index=False)

        ####### basic diagnostic plot
        fig = plt.figure(figsize=[12, 8])
        y_1 = density_pred
        y_2 = density_bin_corrected
        plt.grid(True, linestyle='--', color='grey', linewidth=0.5)

        plt.plot(x_grey, y_1, 'k--', label='raw calibration')
        plt.plot(x_grey, y_2, label='bulk offset calibration')

        Density_list_dic = {
            'air': {'den': 0.001225, 'grey': func_p(0.001225, a, b, c, d), 'color': (211 / 255, 211 / 255, 211 / 255)},
            'epoxy': {'den': 1.13, 'grey': func_p(1.13, a, b, c, d), 'color': (0, 0, 255 / 255)},
            'insert1': {'den': 1.26, 'grey': func_p(1.26, a, b, c, d), 'color': (0, 255 / 255, 0)},
            'insert2': {'den': 1.44, 'grey': func_p(1.44, a, b, c, d), 'color': (255 / 255, 0, 0)},
            'insert3': {'den': 1.65, 'grey': func_p(1.65, a, b, c, d), 'color': (0, 255 / 255, 255 / 255)},
            'insert4': {'den': 1.77, 'grey': func_p(1.77, a, b, c, d), 'color': (255 / 255, 255 / 255, 0)},
            'insert5': {'den': 1.92, 'grey': func_p(1.92, a, b, c, d), 'color': (255 / 255, 0, 255 / 255)},
            'sugar': {'den': 0.1261, 'grey': func_p(0.1261, a, b, c, d), 'color': (0, 0, 128 / 255)},
            'oil': {'den': 0.905, 'grey': func_p(0.904, a, b, c, d), 'color': (0, 128 / 255, 128 / 255)},
            'coffee': {'den': 0.26, 'grey': func_p(0.26, a, b, c, d), 'color': (128 / 255, 128 / 255, 0)},
            'aluminium': {'den': 2.7, 'grey': func_p(2.7, a, b, c, d), 'color': (0, 128 / 255, 0)}}

        for insert in Density_list_dic.keys():
            density_insert = Density_list_dic[insert]['den']
            grey_insert = Density_list_dic[insert]['grey']
            insert_color = Density_list_dic[insert]['color']
            plt.scatter(grey_insert, density_insert, marker='o', color=insert_color,
                        label=f"Standard: {insert}")

        # now plotting the objects and where they fall in the curve
        symbols = ['*', 's', 'v', 'D', '<', '>']
        for k in range(0, len(features_names)):

            plt.scatter(features_avg_grey[k], features_avg_density[k], marker=symbols[k], color=(0, 0, 0), s=100,
                        label=features_names[k])

        plt.legend(loc='upper right', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2)
        plt.ylabel('Density [$\mathregular{g.cm^{-3}}$]')
        plt.xlabel('Grey Value [0:$\mathregular{(2^{16}-1)}$]')
        plt.rcParams["font.family"] = "Arial"
        plt.savefig(os.path.join(scan_folder, 'Diagnostic_Plots_Specimen_' + os.path.basename(scan_folder) + '.png'), dpi=300)

if __name__ == '__main__':

    if platform == "darwin":
        print('This is a Mac')
        multiprocessing.set_start_method(
            'fork')  # Changing this to "fork" (on MAc platforms) otherwise miltiprocessing won't run

    startTime = time.time()

    folder_list, project_folder, calib_dir = find_folders_by_filetype(target_file_type='xtekVolume')

    project_dir_list = []
    project_dir_list.append(project_folder)

    iterable = []
    for each in folder_list:
        iterable.append([each, calib_dir, project_dir_list])

    if len(folder_list) != 0:
        with multiprocessing.Pool(processes=40) as p:
            p.starmap(save_weights_extended_case, iterable)
    else:
        ('All Scans have had their weight estimates extracted')

    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
