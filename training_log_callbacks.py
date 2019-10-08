#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:39:33 2019

@author: pedro
"""
from datetime import datetime
from shutil import copy
import os

from tensorflow.keras.callbacks import Callback
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def create_folder(folder_path):

    # =============================================================================
    #     Creates a folder, if it is not already created
    # =============================================================================

    # Checks if training folder exists before creating it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def custom_pause(interval):

    backend = plt.rcParams['backend']

    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()

        if figManager is not None:
            canvas = figManager.canvas

            if canvas.figure.stale:
                canvas.draw()

            canvas.start_event_loop(interval)

            return


class CallbackSaveLogs(Callback):

    # =============================================================================
    # Saves the configuration file and a training log table
    # =============================================================================

    def __init__(self, folder_path='Training Plots/Training...'):

        self.folder_path = folder_path

    def on_train_begin(self, logs={}):

        create_folder(self.folder_path)

        # copies the configuration file to the training folder
        copy('config.py', self.folder_path + '/' + 'config.txt')

        # dictionary that contains last values for monitored variables
        self.last_value = {}

        # gets timestamp at start of training
        self.timestamp_start = datetime.now()

    def on_epoch_end(self, epoch, logs={}):

        # updates the last value for losses and metrics at the end of each epoch
        for variable, value in logs.items():
            self.last_value[variable] = value

    def on_train_end(self, logs={}):

        # gets timestamp at end of training
        timestamp_end = datetime.now()

        # computes the time spend in training
        delta = timestamp_end - self.timestamp_start

        # extract hours, minutes and seconds
        hours, remainder = divmod(delta.total_seconds(), 60*60)
        minutes, remainder = divmod(remainder, 60)
        seconds, _ = divmod(remainder, 1)

        # creates a pandas series of training logs
        log_series = pd.Series(self.last_value)

        # adds training time to the log series
        log_series['training_time'] = '{:02}:{:02}:{:02}'.format(
            int(hours), int(minutes), int(seconds))

        # saves the training data to a csv file
        log_series.to_csv(self.folder_path + '/' +
                          'Training logs.csv', header=False)


class CallbackPlot(Callback):

    # =============================================================================
    #     Plots the passed losses and metrics, updating them at each
    #     epoch, and saving them at the end
    # =============================================================================
    def __init__(self, plots_settings, title,
                 folder_path='Training Plots/Training...',
                 share_x=False):
        # =============================================================================
        #         Initializes figure
        #         plot_settings: tuple containing dictionaries. Each dictionary corresponds to settings of a plot
        #         title: title of the figure
        #         folder_path: path of folder that will contain all the training information
        #         share_x: either to share the X axis or not
        # =============================================================================

        super().__init__()
        self.plots_settings = plots_settings
        self.plot_count = len(plots_settings)
        self.title = title
        self.share_x = share_x
        self.folder_path = folder_path

    def on_train_begin(self, logs={}):
        plt.ion()
        # creates folder
        create_folder(self.folder_path)

        self.figure, self.windows = plt.subplots(self.plot_count, 1, figsize=[15, 4*self.plot_count], clear=True, num=self.title,
                                                 sharex=self.share_x, constrained_layout=True)

        plt.pause(0.05)
        # dictionary that contains losses and metrics throughout training
        self.losses_and_metrics_dict = {}

        for plot_settings in self.plots_settings:
            for variable in plot_settings['variables'].keys():
                self.losses_and_metrics_dict[variable] = []

    def on_epoch_end(self, epoch, logs={}):

        variables_list = self.losses_and_metrics_dict.keys()

        for variable in variables_list:
            self.losses_and_metrics_dict[variable].append(logs.get(variable))

        # calls the right figure to modify it
        plt.figure(self.title)

        if epoch > 1:

            for window, plot_settings in zip(self.windows, self.plots_settings):

                # clears plot
                window.clear()
                # plt.pause(0.05)

                # checks if the whole data is to be ploted
                if (plot_settings['last_50'] == False) or (epoch < 50):

                    # plots all the variables for this plot
                    for variable, legend in plot_settings['variables'].items():
                        window.plot(
                            self.losses_and_metrics_dict[variable], label=legend)

                    window.set(xlabel='Epoch', ylabel=plot_settings['ylabel'])
                    window.set_title(plot_settings['title'])

                else:

                     # plots all the variables for this plot
                    for variable, legend in plot_settings['variables'].items():
                        window.plot(range(
                            epoch - 50, epoch), self.losses_and_metrics_dict[variable][-50:], label=legend)

                    window.set(xlabel='Last 50 Epochs',
                               ylabel=plot_settings['ylabel'])
                    window.set_title(
                        plot_settings['title'] + ' on last 50 epochs')

                window.legend()
                custom_pause(0.05)

    def on_train_end(self, logs={}):

        # saves losses abd metrics plot
        plt.figure(self.title, clear=False)
        plt.savefig(self.folder_path + '/' + self.title + '.png')
        plt.pause(0.05)
