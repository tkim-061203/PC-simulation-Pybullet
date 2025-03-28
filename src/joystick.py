#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from evdev import InputDevice, categorize, ecodes
from select import select
import numpy as np

class Joystick:
    def __init__(self , event):
        # python3 -m evdev.evtest for identify event
        """
        Initialize the Joystick class with the specified event input device.

        Parameters:
        event (str): The event file path for the input device.

        Attributes:
        gamepad (InputDevice): The input device for the joystick.
        L3 (np.array): Left joystick position as a 2D numpy array.
        R3 (np.array): Right joystick position as a 2D numpy array.
        x (int): Placeholder for 'x' button state.
        triangle (int): Placeholder for 'triangle' button state.
        circle (int): Placeholder for 'circle' button state.
        square (int): Placeholder for 'square' button state.
        T (float): Time duration or interval for certain actions.
        compliantMode (bool): Flag indicating compliant mode state.
        CoM_move (np.array): Center of Mass movement as a 3D numpy array.
        """
        self.gamepad = InputDevice(event)
        self.L3 = np.array([0. , 0.])
        self.R3 = np.array([0. , 0.])
        
        self.x=0
        self.triangle=0
        self.circle=0
        self.square=0
        
        self.T = 0.4
        self.compliantMode = False
        self.CoM_move = np.zeros(3)
    def read(self):
        r,w,x = select([self.gamepad.fd], [], [], 0.)
        
        if r:
            for event in self.gamepad.read():
#                print(event)
                if event.type == ecodes.EV_KEY:
                    if event.value == 1:
                        if event.code == 544:#up arrow
                            self.T += 0.05
                        if event.code == 545:#down arrow
                            self.T -= 0.05
                        if event.code == 308:#square
                            if self.compliantMode == True:
                                self.compliantMode = False
                            elif self.compliantMode == False:
                                self.compliantMode = True    
                        if event.code == 310:#R1
                            self.CoM_move[0] += 0.0005
                        if event.code == 311:#L1
                            self.CoM_move[0] -= 0.0005
                    else:
                        print("boton soltado")
                ########################################  for my own joystick
                #      ^           #     ^            #
                #    ABS_Y         #    ABS_RY        #
                #  ←─────→ ABS_X #  ←─────→ ABS_RX   #
                #     ↓           #     ↓            #  
                #######################################
                elif event.type == ecodes.EV_ABS:
                    absevent = categorize(event)
                    if ecodes.bytype[absevent.event.type][absevent.event.code] == "ABS_X":  
                        self.L3[0]=absevent.event.value-127
                    elif ecodes.bytype[absevent.event.type][absevent.event.code] == "ABS_Y":
                        self.L3[1]=absevent.event.value-127
                    elif ecodes.bytype[absevent.event.type][absevent.event.code] == "ABS_RX":
                        self.R3[0]=absevent.event.value-127
#                        print(self.d_z)
                    elif ecodes.bytype[absevent.event.type][absevent.event.code] == "ABS_RY":
                        self.R3[1]=absevent.event.value-127
                        
        L = np.sqrt(self.L3[1]**2 + self.L3[0]**2)/250.
        angle = np.rad2deg(np.arctan2(-self.L3[0] , -self.L3[1]))
        Lrot = -self.R3[0]/250.
#        Lrot = 0.
        if L <= 0.035:
            L = 0.
        if Lrot <= 0.035 and Lrot >= -0.035:
            Lrot = 0.
            
#        pitch = np.deg2rad(self.R3[1]/2)
#        yaw = np.deg2rad(self.R3[0]/3)
        pitch = 0.
        yaw = 0.
        return self.CoM_move , L , -angle , -Lrot , self.T , self.compliantMode , yaw , pitch
