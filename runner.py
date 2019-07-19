#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:25:57 2019

@author: pedro
"""
'''
plot.clear()
plot = plt.figure(1)
plt.plot(range(10, 20))
'''

import matplotlib.animation as animation
import matplotlib.pyplot as plt


class Example:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def update(self, i):
        print("This is getting called {}".format(i))
        self.ax.plot([i,i+1],[i,i+2])

    def animate(self):
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1000)


def main():
    obj = Example()
    obj.animate()
    plt.show()
    
    obj2 = Example()
    obj2.animate()
    plt.show()


if __name__ == "__main__":
    main()
    
#%%
import matplotlib.pyplot as plt
import numpy as np
import time

# use ggplot style for more sophisticated visuals
plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        global fig
        fig = plt.figure(figsize=(13,6))
        print('olar')
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    #plt.pause(pause_time)
    fig.canvas.flush_events()
    time.sleep(1)
    # return line so we can update it again in the next iteration
    return line1

size = 100
x_vec = np.linspace(0,1,size+1)[0:-1]
y_vec = np.random.randn(len(x_vec))
line1 = []
i=0
while i<1000:
    i=+1
    rand_val = np.random.randn(1)
    y_vec[-1] = rand_val
    line1 = live_plotter(x_vec,y_vec,line1)
    y_vec = np.append(y_vec[1:],0.0)
    
#%%
        
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10*np.pi, 100)
y = np.sin(x)

plt.ion()
fig = plt.figure()

plt.pause(3)
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')

for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    fig.canvas.draw()
