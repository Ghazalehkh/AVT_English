#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on April 04, 2024, at 16:34
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'mYousefi'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\alamol\\Desktop\\M_Yousefi\\mYousefi.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Start" ---
    back8 = visual.Rect(
        win=win, name='back8',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    practiceText = visual.TextStim(win=win, name='practiceText',
        text="Press 'space' to start the practice trial...",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Resp4Space = keyboard.Keyboard()
    
    # --- Initialize components for Routine "taskT1" ---
    back3 = visual.Rect(
        win=win, name='back3',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    OtherText = visual.TextStim(win=win, name='OtherText',
        text='',
        font='Open Sans',
        units='norm', pos=(0,0.75), height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Pic = visual.ImageStim(
        win=win,
        name='Pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    SoundRout = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='SoundRout')
    SoundRout.setVolume(1.0)
    
    # --- Initialize components for Routine "Response" ---
    back6 = visual.Rect(
        win=win, name='back6',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    Quest = visual.TextStim(win=win, name='Quest',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    RatingResp = visual.Slider(win=win, name='RatingResp',
        startValue=None, size=(1.65, 0.1), pos=(0, -0.3), units=win.units,
        labels=[-10 ,0 , +10], ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=1.0,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixationCross = visual.ShapeStim(
        win=win, name='FixationCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "instruction" ---
    background = visual.Rect(
        win=win, name='background',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    PressText = visual.TextStim(win=win, name='PressText',
        text="Press 'space' if you are ready...",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Resp1Space = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixationCross = visual.ShapeStim(
        win=win, name='FixationCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "taskT1" ---
    back3 = visual.Rect(
        win=win, name='back3',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    OtherText = visual.TextStim(win=win, name='OtherText',
        text='',
        font='Open Sans',
        units='norm', pos=(0,0.75), height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Pic = visual.ImageStim(
        win=win,
        name='Pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    SoundRout = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='SoundRout')
    SoundRout.setVolume(1.0)
    
    # --- Initialize components for Routine "Response" ---
    back6 = visual.Rect(
        win=win, name='back6',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    Quest = visual.TextStim(win=win, name='Quest',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    RatingResp = visual.Slider(win=win, name='RatingResp',
        startValue=None, size=(1.65, 0.1), pos=(0, -0.3), units=win.units,
        labels=[-10 ,0 , +10], ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=1.0,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixationCross = visual.ShapeStim(
        win=win, name='FixationCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "BreakRoutine" ---
    back5 = visual.Rect(
        win=win, name='back5',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    BreakText = visual.TextStim(win=win, name='BreakText',
        text="Have a break...\npress 'space' if you are ready",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    Resp2Space = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixationCross = visual.ShapeStim(
        win=win, name='FixationCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "taskT1" ---
    back3 = visual.Rect(
        win=win, name='back3',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    OtherText = visual.TextStim(win=win, name='OtherText',
        text='',
        font='Open Sans',
        units='norm', pos=(0,0.75), height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Pic = visual.ImageStim(
        win=win,
        name='Pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0,0), size=(1,1),
        color=[1,1,1], colorSpace='rgb', opacity=1.0,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    SoundRout = sound.Sound('A', secs=-1, stereo=True, hamming=True,
        name='SoundRout')
    SoundRout.setVolume(1.0)
    
    # --- Initialize components for Routine "Response" ---
    back6 = visual.Rect(
        win=win, name='back6',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    Quest = visual.TextStim(win=win, name='Quest',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    RatingResp = visual.Slider(win=win, name='RatingResp',
        startValue=None, size=(1.65, 0.1), pos=(0, -0.3), units=win.units,
        labels=[-10 ,0 , +10], ticks=(-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1, 2, 3, 4, 5,6,7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=1.0,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    
    # --- Initialize components for Routine "Fc" ---
    back2 = visual.Rect(
        win=win, name='back2',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    FixationCross = visual.ShapeStim(
        win=win, name='FixationCross', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=-1.0, interpolate=True)
    
    # --- Initialize components for Routine "Finish" ---
    back7 = visual.Rect(
        win=win, name='back7',units='norm', 
        width=(2,2)[0], height=(2,2)[1],
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='black', fillColor='black',
        opacity=1.0, depth=0.0, interpolate=True)
    TnxParticipants = visual.TextStim(win=win, name='TnxParticipants',
        text='Thanks for your Particiption!',
        font='Open Sans',
        pos=(0, 0), height=0.09, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-1.0);
    Resp3Space = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Start" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Start.started', globalClock.getTime())
    Resp4Space.keys = []
    Resp4Space.rt = []
    _Resp4Space_allKeys = []
    # keep track of which components have finished
    StartComponents = [back8, practiceText, Resp4Space]
    for thisComponent in StartComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back8* updates
        
        # if back8 is starting this frame...
        if back8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back8.frameNStart = frameN  # exact frame index
            back8.tStart = t  # local t and not account for scr refresh
            back8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'back8.started')
            # update status
            back8.status = STARTED
            back8.setAutoDraw(True)
        
        # if back8 is active this frame...
        if back8.status == STARTED:
            # update params
            pass
        
        # *practiceText* updates
        
        # if practiceText is starting this frame...
        if practiceText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practiceText.frameNStart = frameN  # exact frame index
            practiceText.tStart = t  # local t and not account for scr refresh
            practiceText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practiceText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practiceText.started')
            # update status
            practiceText.status = STARTED
            practiceText.setAutoDraw(True)
        
        # if practiceText is active this frame...
        if practiceText.status == STARTED:
            # update params
            pass
        
        # *Resp4Space* updates
        waitOnFlip = False
        
        # if Resp4Space is starting this frame...
        if Resp4Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp4Space.frameNStart = frameN  # exact frame index
            Resp4Space.tStart = t  # local t and not account for scr refresh
            Resp4Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp4Space, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Resp4Space.started')
            # update status
            Resp4Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp4Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp4Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp4Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp4Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp4Space_allKeys.extend(theseKeys)
            if len(_Resp4Space_allKeys):
                Resp4Space.keys = _Resp4Space_allKeys[-1].name  # just the last key pressed
                Resp4Space.rt = _Resp4Space_allKeys[-1].rt
                Resp4Space.duration = _Resp4Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in StartComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start" ---
    for thisComponent in StartComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Start.stopped', globalClock.getTime())
    # the Routine "Start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    PracticeTrial = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('practice.xlsx'),
        seed=None, name='PracticeTrial')
    thisExp.addLoop(PracticeTrial)  # add the loop to the experiment
    thisPracticeTrial = PracticeTrial.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPracticeTrial.rgb)
    if thisPracticeTrial != None:
        for paramName in thisPracticeTrial:
            globals()[paramName] = thisPracticeTrial[paramName]
    
    for thisPracticeTrial in PracticeTrial:
        currentLoop = PracticeTrial
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPracticeTrial.rgb)
        if thisPracticeTrial != None:
            for paramName in thisPracticeTrial:
                globals()[paramName] = thisPracticeTrial[paramName]
        
        # --- Prepare to start Routine "taskT1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('taskT1.started', globalClock.getTime())
        OtherText.setText('The Other Person is listening to...')
        Pic.setImage(image)
        SoundRout.setSound(vc, secs=duration, hamming=True)
        SoundRout.setVolume(1.0, log=False)
        SoundRout.seek(0)
        # keep track of which components have finished
        taskT1Components = [back3, OtherText, Pic, SoundRout]
        for thisComponent in taskT1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskT1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back3* updates
            
            # if back3 is starting this frame...
            if back3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back3.frameNStart = frameN  # exact frame index
                back3.tStart = t  # local t and not account for scr refresh
                back3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back3, 'tStartRefresh')  # time at next scr refresh
                # update status
                back3.status = STARTED
                back3.setAutoDraw(True)
            
            # if back3 is active this frame...
            if back3.status == STARTED:
                # update params
                pass
            
            # if back3 is stopping this frame...
            if back3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back3.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    back3.tStop = t  # not accounting for scr refresh
                    back3.frameNStop = frameN  # exact frame index
                    # update status
                    back3.status = FINISHED
                    back3.setAutoDraw(False)
            
            # *OtherText* updates
            
            # if OtherText is starting this frame...
            if OtherText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                OtherText.frameNStart = frameN  # exact frame index
                OtherText.tStart = t  # local t and not account for scr refresh
                OtherText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(OtherText, 'tStartRefresh')  # time at next scr refresh
                # update status
                OtherText.status = STARTED
                OtherText.setAutoDraw(True)
            
            # if OtherText is active this frame...
            if OtherText.status == STARTED:
                # update params
                pass
            
            # if OtherText is stopping this frame...
            if OtherText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > OtherText.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    OtherText.tStop = t  # not accounting for scr refresh
                    OtherText.frameNStop = frameN  # exact frame index
                    # update status
                    OtherText.status = FINISHED
                    OtherText.setAutoDraw(False)
            
            # *Pic* updates
            
            # if Pic is starting this frame...
            if Pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pic.frameNStart = frameN  # exact frame index
                Pic.tStart = t  # local t and not account for scr refresh
                Pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                Pic.status = STARTED
                Pic.setAutoDraw(True)
            
            # if Pic is active this frame...
            if Pic.status == STARTED:
                # update params
                pass
            
            # if Pic is stopping this frame...
            if Pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pic.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Pic.tStop = t  # not accounting for scr refresh
                    Pic.frameNStop = frameN  # exact frame index
                    # update status
                    Pic.status = FINISHED
                    Pic.setAutoDraw(False)
            
            # if SoundRout is starting this frame...
            if SoundRout.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                SoundRout.frameNStart = frameN  # exact frame index
                SoundRout.tStart = t  # local t and not account for scr refresh
                SoundRout.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                SoundRout.status = STARTED
                SoundRout.play(when=win)  # sync with win flip
            
            # if SoundRout is stopping this frame...
            if SoundRout.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > SoundRout.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    SoundRout.tStop = t  # not accounting for scr refresh
                    SoundRout.frameNStop = frameN  # exact frame index
                    # update status
                    SoundRout.status = FINISHED
                    SoundRout.stop()
            # update SoundRout status according to whether it's playing
            if SoundRout.isPlaying:
                SoundRout.status = STARTED
            elif SoundRout.isFinished:
                SoundRout.status = FINISHED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskT1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskT1" ---
        for thisComponent in taskT1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('taskT1.stopped', globalClock.getTime())
        SoundRout.pause()  # ensure sound has stopped at end of Routine
        # the Routine "taskT1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Response" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Response.started', globalClock.getTime())
        Quest.setText(text)
        RatingResp.reset()
        # keep track of which components have finished
        ResponseComponents = [back6, Quest, RatingResp]
        for thisComponent in ResponseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back6* updates
            
            # if back6 is starting this frame...
            if back6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back6.frameNStart = frameN  # exact frame index
                back6.tStart = t  # local t and not account for scr refresh
                back6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'back6.started')
                # update status
                back6.status = STARTED
                back6.setAutoDraw(True)
            
            # if back6 is active this frame...
            if back6.status == STARTED:
                # update params
                pass
            
            # if back6 is stopping this frame...
            if back6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    back6.tStop = t  # not accounting for scr refresh
                    back6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back6.stopped')
                    # update status
                    back6.status = FINISHED
                    back6.setAutoDraw(False)
            
            # *Quest* updates
            
            # if Quest is starting this frame...
            if Quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Quest.frameNStart = frameN  # exact frame index
                Quest.tStart = t  # local t and not account for scr refresh
                Quest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quest.started')
                # update status
                Quest.status = STARTED
                Quest.setAutoDraw(True)
            
            # if Quest is active this frame...
            if Quest.status == STARTED:
                # update params
                pass
            
            # if Quest is stopping this frame...
            if Quest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Quest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Quest.tStop = t  # not accounting for scr refresh
                    Quest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Quest.stopped')
                    # update status
                    Quest.status = FINISHED
                    Quest.setAutoDraw(False)
            
            # *RatingResp* updates
            
            # if RatingResp is starting this frame...
            if RatingResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RatingResp.frameNStart = frameN  # exact frame index
                RatingResp.tStart = t  # local t and not account for scr refresh
                RatingResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RatingResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RatingResp.started')
                # update status
                RatingResp.status = STARTED
                RatingResp.setAutoDraw(True)
            
            # if RatingResp is active this frame...
            if RatingResp.status == STARTED:
                # update params
                pass
            
            # if RatingResp is stopping this frame...
            if RatingResp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > RatingResp.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    RatingResp.tStop = t  # not accounting for scr refresh
                    RatingResp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RatingResp.stopped')
                    # update status
                    RatingResp.status = FINISHED
                    RatingResp.setAutoDraw(False)
            
            # Check RatingResp for response to end Routine
            if RatingResp.getRating() is not None and RatingResp.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ResponseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in ResponseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Response.stopped', globalClock.getTime())
        PracticeTrial.addData('RatingResp.response', RatingResp.getRating())
        PracticeTrial.addData('RatingResp.rt', RatingResp.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "Fc" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fc.started', globalClock.getTime())
        # keep track of which components have finished
        FcComponents = [back2, FixationCross]
        for thisComponent in FcComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fc" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back2* updates
            
            # if back2 is starting this frame...
            if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back2.frameNStart = frameN  # exact frame index
                back2.tStart = t  # local t and not account for scr refresh
                back2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
                # update status
                back2.status = STARTED
                back2.setAutoDraw(True)
            
            # if back2 is active this frame...
            if back2.status == STARTED:
                # update params
                pass
            
            # if back2 is stopping this frame...
            if back2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    back2.tStop = t  # not accounting for scr refresh
                    back2.frameNStop = frameN  # exact frame index
                    # update status
                    back2.status = FINISHED
                    back2.setAutoDraw(False)
            
            # *FixationCross* updates
            
            # if FixationCross is starting this frame...
            if FixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCross.frameNStart = frameN  # exact frame index
                FixationCross.tStart = t  # local t and not account for scr refresh
                FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
                # update status
                FixationCross.status = STARTED
                FixationCross.setAutoDraw(True)
            
            # if FixationCross is active this frame...
            if FixationCross.status == STARTED:
                # update params
                pass
            
            # if FixationCross is stopping this frame...
            if FixationCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCross.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCross.tStop = t  # not accounting for scr refresh
                    FixationCross.frameNStop = frameN  # exact frame index
                    # update status
                    FixationCross.status = FINISHED
                    FixationCross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FcComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fc" ---
        for thisComponent in FcComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fc.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
    # completed 1.0 repeats of 'PracticeTrial'
    
    
    # --- Prepare to start Routine "instruction" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruction.started', globalClock.getTime())
    Resp1Space.keys = []
    Resp1Space.rt = []
    _Resp1Space_allKeys = []
    # keep track of which components have finished
    instructionComponents = [background, PressText, Resp1Space]
    for thisComponent in instructionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *background* updates
        
        # if background is starting this frame...
        if background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            background.frameNStart = frameN  # exact frame index
            background.tStart = t  # local t and not account for scr refresh
            background.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(background, 'tStartRefresh')  # time at next scr refresh
            # update status
            background.status = STARTED
            background.setAutoDraw(True)
        
        # if background is active this frame...
        if background.status == STARTED:
            # update params
            pass
        
        # *PressText* updates
        
        # if PressText is starting this frame...
        if PressText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            PressText.frameNStart = frameN  # exact frame index
            PressText.tStart = t  # local t and not account for scr refresh
            PressText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(PressText, 'tStartRefresh')  # time at next scr refresh
            # update status
            PressText.status = STARTED
            PressText.setAutoDraw(True)
        
        # if PressText is active this frame...
        if PressText.status == STARTED:
            # update params
            pass
        
        # *Resp1Space* updates
        waitOnFlip = False
        
        # if Resp1Space is starting this frame...
        if Resp1Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp1Space.frameNStart = frameN  # exact frame index
            Resp1Space.tStart = t  # local t and not account for scr refresh
            Resp1Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp1Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp1Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp1Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp1Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp1Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp1Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp1Space_allKeys.extend(theseKeys)
            if len(_Resp1Space_allKeys):
                Resp1Space.keys = _Resp1Space_allKeys[-1].name  # just the last key pressed
                Resp1Space.rt = _Resp1Space_allKeys[-1].rt
                Resp1Space.duration = _Resp1Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruction.stopped', globalClock.getTime())
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Fc" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Fc.started', globalClock.getTime())
    # keep track of which components have finished
    FcComponents = [back2, FixationCross]
    for thisComponent in FcComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Fc" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back2* updates
        
        # if back2 is starting this frame...
        if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back2.frameNStart = frameN  # exact frame index
            back2.tStart = t  # local t and not account for scr refresh
            back2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
            # update status
            back2.status = STARTED
            back2.setAutoDraw(True)
        
        # if back2 is active this frame...
        if back2.status == STARTED:
            # update params
            pass
        
        # if back2 is stopping this frame...
        if back2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                back2.tStop = t  # not accounting for scr refresh
                back2.frameNStop = frameN  # exact frame index
                # update status
                back2.status = FINISHED
                back2.setAutoDraw(False)
        
        # *FixationCross* updates
        
        # if FixationCross is starting this frame...
        if FixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            FixationCross.frameNStart = frameN  # exact frame index
            FixationCross.tStart = t  # local t and not account for scr refresh
            FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
            # update status
            FixationCross.status = STARTED
            FixationCross.setAutoDraw(True)
        
        # if FixationCross is active this frame...
        if FixationCross.status == STARTED:
            # update params
            pass
        
        # if FixationCross is stopping this frame...
        if FixationCross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > FixationCross.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                FixationCross.tStop = t  # not accounting for scr refresh
                FixationCross.frameNStop = frameN  # exact frame index
                # update status
                FixationCross.status = FINISHED
                FixationCross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in FcComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fc" ---
    for thisComponent in FcComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Fc.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # set up handler to look after randomisation of conditions etc
    Try1 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('before.xlsx'),
        seed=None, name='Try1')
    thisExp.addLoop(Try1)  # add the loop to the experiment
    thisTry1 = Try1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTry1.rgb)
    if thisTry1 != None:
        for paramName in thisTry1:
            globals()[paramName] = thisTry1[paramName]
    
    for thisTry1 in Try1:
        currentLoop = Try1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTry1.rgb)
        if thisTry1 != None:
            for paramName in thisTry1:
                globals()[paramName] = thisTry1[paramName]
        
        # --- Prepare to start Routine "taskT1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('taskT1.started', globalClock.getTime())
        OtherText.setText('The Other Person is listening to...')
        Pic.setImage(image)
        SoundRout.setSound(vc, secs=duration, hamming=True)
        SoundRout.setVolume(1.0, log=False)
        SoundRout.seek(0)
        # keep track of which components have finished
        taskT1Components = [back3, OtherText, Pic, SoundRout]
        for thisComponent in taskT1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskT1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back3* updates
            
            # if back3 is starting this frame...
            if back3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back3.frameNStart = frameN  # exact frame index
                back3.tStart = t  # local t and not account for scr refresh
                back3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back3, 'tStartRefresh')  # time at next scr refresh
                # update status
                back3.status = STARTED
                back3.setAutoDraw(True)
            
            # if back3 is active this frame...
            if back3.status == STARTED:
                # update params
                pass
            
            # if back3 is stopping this frame...
            if back3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back3.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    back3.tStop = t  # not accounting for scr refresh
                    back3.frameNStop = frameN  # exact frame index
                    # update status
                    back3.status = FINISHED
                    back3.setAutoDraw(False)
            
            # *OtherText* updates
            
            # if OtherText is starting this frame...
            if OtherText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                OtherText.frameNStart = frameN  # exact frame index
                OtherText.tStart = t  # local t and not account for scr refresh
                OtherText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(OtherText, 'tStartRefresh')  # time at next scr refresh
                # update status
                OtherText.status = STARTED
                OtherText.setAutoDraw(True)
            
            # if OtherText is active this frame...
            if OtherText.status == STARTED:
                # update params
                pass
            
            # if OtherText is stopping this frame...
            if OtherText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > OtherText.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    OtherText.tStop = t  # not accounting for scr refresh
                    OtherText.frameNStop = frameN  # exact frame index
                    # update status
                    OtherText.status = FINISHED
                    OtherText.setAutoDraw(False)
            
            # *Pic* updates
            
            # if Pic is starting this frame...
            if Pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pic.frameNStart = frameN  # exact frame index
                Pic.tStart = t  # local t and not account for scr refresh
                Pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                Pic.status = STARTED
                Pic.setAutoDraw(True)
            
            # if Pic is active this frame...
            if Pic.status == STARTED:
                # update params
                pass
            
            # if Pic is stopping this frame...
            if Pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pic.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Pic.tStop = t  # not accounting for scr refresh
                    Pic.frameNStop = frameN  # exact frame index
                    # update status
                    Pic.status = FINISHED
                    Pic.setAutoDraw(False)
            
            # if SoundRout is starting this frame...
            if SoundRout.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                SoundRout.frameNStart = frameN  # exact frame index
                SoundRout.tStart = t  # local t and not account for scr refresh
                SoundRout.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                SoundRout.status = STARTED
                SoundRout.play(when=win)  # sync with win flip
            
            # if SoundRout is stopping this frame...
            if SoundRout.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > SoundRout.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    SoundRout.tStop = t  # not accounting for scr refresh
                    SoundRout.frameNStop = frameN  # exact frame index
                    # update status
                    SoundRout.status = FINISHED
                    SoundRout.stop()
            # update SoundRout status according to whether it's playing
            if SoundRout.isPlaying:
                SoundRout.status = STARTED
            elif SoundRout.isFinished:
                SoundRout.status = FINISHED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskT1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskT1" ---
        for thisComponent in taskT1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('taskT1.stopped', globalClock.getTime())
        SoundRout.pause()  # ensure sound has stopped at end of Routine
        # the Routine "taskT1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Response" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Response.started', globalClock.getTime())
        Quest.setText(text)
        RatingResp.reset()
        # keep track of which components have finished
        ResponseComponents = [back6, Quest, RatingResp]
        for thisComponent in ResponseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back6* updates
            
            # if back6 is starting this frame...
            if back6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back6.frameNStart = frameN  # exact frame index
                back6.tStart = t  # local t and not account for scr refresh
                back6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'back6.started')
                # update status
                back6.status = STARTED
                back6.setAutoDraw(True)
            
            # if back6 is active this frame...
            if back6.status == STARTED:
                # update params
                pass
            
            # if back6 is stopping this frame...
            if back6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    back6.tStop = t  # not accounting for scr refresh
                    back6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back6.stopped')
                    # update status
                    back6.status = FINISHED
                    back6.setAutoDraw(False)
            
            # *Quest* updates
            
            # if Quest is starting this frame...
            if Quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Quest.frameNStart = frameN  # exact frame index
                Quest.tStart = t  # local t and not account for scr refresh
                Quest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quest.started')
                # update status
                Quest.status = STARTED
                Quest.setAutoDraw(True)
            
            # if Quest is active this frame...
            if Quest.status == STARTED:
                # update params
                pass
            
            # if Quest is stopping this frame...
            if Quest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Quest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Quest.tStop = t  # not accounting for scr refresh
                    Quest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Quest.stopped')
                    # update status
                    Quest.status = FINISHED
                    Quest.setAutoDraw(False)
            
            # *RatingResp* updates
            
            # if RatingResp is starting this frame...
            if RatingResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RatingResp.frameNStart = frameN  # exact frame index
                RatingResp.tStart = t  # local t and not account for scr refresh
                RatingResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RatingResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RatingResp.started')
                # update status
                RatingResp.status = STARTED
                RatingResp.setAutoDraw(True)
            
            # if RatingResp is active this frame...
            if RatingResp.status == STARTED:
                # update params
                pass
            
            # if RatingResp is stopping this frame...
            if RatingResp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > RatingResp.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    RatingResp.tStop = t  # not accounting for scr refresh
                    RatingResp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RatingResp.stopped')
                    # update status
                    RatingResp.status = FINISHED
                    RatingResp.setAutoDraw(False)
            
            # Check RatingResp for response to end Routine
            if RatingResp.getRating() is not None and RatingResp.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ResponseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in ResponseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Response.stopped', globalClock.getTime())
        Try1.addData('RatingResp.response', RatingResp.getRating())
        Try1.addData('RatingResp.rt', RatingResp.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "Fc" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fc.started', globalClock.getTime())
        # keep track of which components have finished
        FcComponents = [back2, FixationCross]
        for thisComponent in FcComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fc" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back2* updates
            
            # if back2 is starting this frame...
            if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back2.frameNStart = frameN  # exact frame index
                back2.tStart = t  # local t and not account for scr refresh
                back2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
                # update status
                back2.status = STARTED
                back2.setAutoDraw(True)
            
            # if back2 is active this frame...
            if back2.status == STARTED:
                # update params
                pass
            
            # if back2 is stopping this frame...
            if back2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    back2.tStop = t  # not accounting for scr refresh
                    back2.frameNStop = frameN  # exact frame index
                    # update status
                    back2.status = FINISHED
                    back2.setAutoDraw(False)
            
            # *FixationCross* updates
            
            # if FixationCross is starting this frame...
            if FixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCross.frameNStart = frameN  # exact frame index
                FixationCross.tStart = t  # local t and not account for scr refresh
                FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
                # update status
                FixationCross.status = STARTED
                FixationCross.setAutoDraw(True)
            
            # if FixationCross is active this frame...
            if FixationCross.status == STARTED:
                # update params
                pass
            
            # if FixationCross is stopping this frame...
            if FixationCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCross.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCross.tStop = t  # not accounting for scr refresh
                    FixationCross.frameNStop = frameN  # exact frame index
                    # update status
                    FixationCross.status = FINISHED
                    FixationCross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FcComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fc" ---
        for thisComponent in FcComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fc.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'Try1'
    
    
    # --- Prepare to start Routine "BreakRoutine" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('BreakRoutine.started', globalClock.getTime())
    Resp2Space.keys = []
    Resp2Space.rt = []
    _Resp2Space_allKeys = []
    # keep track of which components have finished
    BreakRoutineComponents = [back5, BreakText, Resp2Space]
    for thisComponent in BreakRoutineComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BreakRoutine" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back5* updates
        
        # if back5 is starting this frame...
        if back5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back5.frameNStart = frameN  # exact frame index
            back5.tStart = t  # local t and not account for scr refresh
            back5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back5, 'tStartRefresh')  # time at next scr refresh
            # update status
            back5.status = STARTED
            back5.setAutoDraw(True)
        
        # if back5 is active this frame...
        if back5.status == STARTED:
            # update params
            pass
        
        # *BreakText* updates
        
        # if BreakText is starting this frame...
        if BreakText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            BreakText.frameNStart = frameN  # exact frame index
            BreakText.tStart = t  # local t and not account for scr refresh
            BreakText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(BreakText, 'tStartRefresh')  # time at next scr refresh
            # update status
            BreakText.status = STARTED
            BreakText.setAutoDraw(True)
        
        # if BreakText is active this frame...
        if BreakText.status == STARTED:
            # update params
            pass
        
        # *Resp2Space* updates
        waitOnFlip = False
        
        # if Resp2Space is starting this frame...
        if Resp2Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp2Space.frameNStart = frameN  # exact frame index
            Resp2Space.tStart = t  # local t and not account for scr refresh
            Resp2Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp2Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp2Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp2Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp2Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp2Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp2Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp2Space_allKeys.extend(theseKeys)
            if len(_Resp2Space_allKeys):
                Resp2Space.keys = _Resp2Space_allKeys[-1].name  # just the last key pressed
                Resp2Space.rt = _Resp2Space_allKeys[-1].rt
                Resp2Space.duration = _Resp2Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in BreakRoutineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BreakRoutine" ---
    for thisComponent in BreakRoutineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('BreakRoutine.stopped', globalClock.getTime())
    # the Routine "BreakRoutine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Fc" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Fc.started', globalClock.getTime())
    # keep track of which components have finished
    FcComponents = [back2, FixationCross]
    for thisComponent in FcComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Fc" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back2* updates
        
        # if back2 is starting this frame...
        if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back2.frameNStart = frameN  # exact frame index
            back2.tStart = t  # local t and not account for scr refresh
            back2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
            # update status
            back2.status = STARTED
            back2.setAutoDraw(True)
        
        # if back2 is active this frame...
        if back2.status == STARTED:
            # update params
            pass
        
        # if back2 is stopping this frame...
        if back2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                # keep track of stop time/frame for later
                back2.tStop = t  # not accounting for scr refresh
                back2.frameNStop = frameN  # exact frame index
                # update status
                back2.status = FINISHED
                back2.setAutoDraw(False)
        
        # *FixationCross* updates
        
        # if FixationCross is starting this frame...
        if FixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            FixationCross.frameNStart = frameN  # exact frame index
            FixationCross.tStart = t  # local t and not account for scr refresh
            FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
            # update status
            FixationCross.status = STARTED
            FixationCross.setAutoDraw(True)
        
        # if FixationCross is active this frame...
        if FixationCross.status == STARTED:
            # update params
            pass
        
        # if FixationCross is stopping this frame...
        if FixationCross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > FixationCross.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                FixationCross.tStop = t  # not accounting for scr refresh
                FixationCross.frameNStop = frameN  # exact frame index
                # update status
                FixationCross.status = FINISHED
                FixationCross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in FcComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fc" ---
    for thisComponent in FcComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Fc.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # set up handler to look after randomisation of conditions etc
    Try2 = data.TrialHandler(nReps=1.0, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('after.xlsx'),
        seed=None, name='Try2')
    thisExp.addLoop(Try2)  # add the loop to the experiment
    thisTry2 = Try2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTry2.rgb)
    if thisTry2 != None:
        for paramName in thisTry2:
            globals()[paramName] = thisTry2[paramName]
    
    for thisTry2 in Try2:
        currentLoop = Try2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTry2.rgb)
        if thisTry2 != None:
            for paramName in thisTry2:
                globals()[paramName] = thisTry2[paramName]
        
        # --- Prepare to start Routine "taskT1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('taskT1.started', globalClock.getTime())
        OtherText.setText('The Other Person is listening to...')
        Pic.setImage(image)
        SoundRout.setSound(vc, secs=duration, hamming=True)
        SoundRout.setVolume(1.0, log=False)
        SoundRout.seek(0)
        # keep track of which components have finished
        taskT1Components = [back3, OtherText, Pic, SoundRout]
        for thisComponent in taskT1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "taskT1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back3* updates
            
            # if back3 is starting this frame...
            if back3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back3.frameNStart = frameN  # exact frame index
                back3.tStart = t  # local t and not account for scr refresh
                back3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back3, 'tStartRefresh')  # time at next scr refresh
                # update status
                back3.status = STARTED
                back3.setAutoDraw(True)
            
            # if back3 is active this frame...
            if back3.status == STARTED:
                # update params
                pass
            
            # if back3 is stopping this frame...
            if back3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back3.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    back3.tStop = t  # not accounting for scr refresh
                    back3.frameNStop = frameN  # exact frame index
                    # update status
                    back3.status = FINISHED
                    back3.setAutoDraw(False)
            
            # *OtherText* updates
            
            # if OtherText is starting this frame...
            if OtherText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                OtherText.frameNStart = frameN  # exact frame index
                OtherText.tStart = t  # local t and not account for scr refresh
                OtherText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(OtherText, 'tStartRefresh')  # time at next scr refresh
                # update status
                OtherText.status = STARTED
                OtherText.setAutoDraw(True)
            
            # if OtherText is active this frame...
            if OtherText.status == STARTED:
                # update params
                pass
            
            # if OtherText is stopping this frame...
            if OtherText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > OtherText.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    OtherText.tStop = t  # not accounting for scr refresh
                    OtherText.frameNStop = frameN  # exact frame index
                    # update status
                    OtherText.status = FINISHED
                    OtherText.setAutoDraw(False)
            
            # *Pic* updates
            
            # if Pic is starting this frame...
            if Pic.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Pic.frameNStart = frameN  # exact frame index
                Pic.tStart = t  # local t and not account for scr refresh
                Pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Pic, 'tStartRefresh')  # time at next scr refresh
                # update status
                Pic.status = STARTED
                Pic.setAutoDraw(True)
            
            # if Pic is active this frame...
            if Pic.status == STARTED:
                # update params
                pass
            
            # if Pic is stopping this frame...
            if Pic.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Pic.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    Pic.tStop = t  # not accounting for scr refresh
                    Pic.frameNStop = frameN  # exact frame index
                    # update status
                    Pic.status = FINISHED
                    Pic.setAutoDraw(False)
            
            # if SoundRout is starting this frame...
            if SoundRout.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                SoundRout.frameNStart = frameN  # exact frame index
                SoundRout.tStart = t  # local t and not account for scr refresh
                SoundRout.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                SoundRout.status = STARTED
                SoundRout.play(when=win)  # sync with win flip
            
            # if SoundRout is stopping this frame...
            if SoundRout.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > SoundRout.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    SoundRout.tStop = t  # not accounting for scr refresh
                    SoundRout.frameNStop = frameN  # exact frame index
                    # update status
                    SoundRout.status = FINISHED
                    SoundRout.stop()
            # update SoundRout status according to whether it's playing
            if SoundRout.isPlaying:
                SoundRout.status = STARTED
            elif SoundRout.isFinished:
                SoundRout.status = FINISHED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in taskT1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "taskT1" ---
        for thisComponent in taskT1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('taskT1.stopped', globalClock.getTime())
        SoundRout.pause()  # ensure sound has stopped at end of Routine
        # the Routine "taskT1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Response" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Response.started', globalClock.getTime())
        Quest.setText(text)
        RatingResp.reset()
        # keep track of which components have finished
        ResponseComponents = [back6, Quest, RatingResp]
        for thisComponent in ResponseComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Response" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back6* updates
            
            # if back6 is starting this frame...
            if back6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back6.frameNStart = frameN  # exact frame index
                back6.tStart = t  # local t and not account for scr refresh
                back6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'back6.started')
                # update status
                back6.status = STARTED
                back6.setAutoDraw(True)
            
            # if back6 is active this frame...
            if back6.status == STARTED:
                # update params
                pass
            
            # if back6 is stopping this frame...
            if back6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    back6.tStop = t  # not accounting for scr refresh
                    back6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'back6.stopped')
                    # update status
                    back6.status = FINISHED
                    back6.setAutoDraw(False)
            
            # *Quest* updates
            
            # if Quest is starting this frame...
            if Quest.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Quest.frameNStart = frameN  # exact frame index
                Quest.tStart = t  # local t and not account for scr refresh
                Quest.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Quest, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Quest.started')
                # update status
                Quest.status = STARTED
                Quest.setAutoDraw(True)
            
            # if Quest is active this frame...
            if Quest.status == STARTED:
                # update params
                pass
            
            # if Quest is stopping this frame...
            if Quest.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Quest.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    Quest.tStop = t  # not accounting for scr refresh
                    Quest.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Quest.stopped')
                    # update status
                    Quest.status = FINISHED
                    Quest.setAutoDraw(False)
            
            # *RatingResp* updates
            
            # if RatingResp is starting this frame...
            if RatingResp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                RatingResp.frameNStart = frameN  # exact frame index
                RatingResp.tStart = t  # local t and not account for scr refresh
                RatingResp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(RatingResp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'RatingResp.started')
                # update status
                RatingResp.status = STARTED
                RatingResp.setAutoDraw(True)
            
            # if RatingResp is active this frame...
            if RatingResp.status == STARTED:
                # update params
                pass
            
            # if RatingResp is stopping this frame...
            if RatingResp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > RatingResp.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    RatingResp.tStop = t  # not accounting for scr refresh
                    RatingResp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'RatingResp.stopped')
                    # update status
                    RatingResp.status = FINISHED
                    RatingResp.setAutoDraw(False)
            
            # Check RatingResp for response to end Routine
            if RatingResp.getRating() is not None and RatingResp.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ResponseComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Response" ---
        for thisComponent in ResponseComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Response.stopped', globalClock.getTime())
        Try2.addData('RatingResp.response', RatingResp.getRating())
        Try2.addData('RatingResp.rt', RatingResp.getRT())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "Fc" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Fc.started', globalClock.getTime())
        # keep track of which components have finished
        FcComponents = [back2, FixationCross]
        for thisComponent in FcComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Fc" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *back2* updates
            
            # if back2 is starting this frame...
            if back2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                back2.frameNStart = frameN  # exact frame index
                back2.tStart = t  # local t and not account for scr refresh
                back2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(back2, 'tStartRefresh')  # time at next scr refresh
                # update status
                back2.status = STARTED
                back2.setAutoDraw(True)
            
            # if back2 is active this frame...
            if back2.status == STARTED:
                # update params
                pass
            
            # if back2 is stopping this frame...
            if back2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > back2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    back2.tStop = t  # not accounting for scr refresh
                    back2.frameNStop = frameN  # exact frame index
                    # update status
                    back2.status = FINISHED
                    back2.setAutoDraw(False)
            
            # *FixationCross* updates
            
            # if FixationCross is starting this frame...
            if FixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                FixationCross.frameNStart = frameN  # exact frame index
                FixationCross.tStart = t  # local t and not account for scr refresh
                FixationCross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(FixationCross, 'tStartRefresh')  # time at next scr refresh
                # update status
                FixationCross.status = STARTED
                FixationCross.setAutoDraw(True)
            
            # if FixationCross is active this frame...
            if FixationCross.status == STARTED:
                # update params
                pass
            
            # if FixationCross is stopping this frame...
            if FixationCross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > FixationCross.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    FixationCross.tStop = t  # not accounting for scr refresh
                    FixationCross.frameNStop = frameN  # exact frame index
                    # update status
                    FixationCross.status = FINISHED
                    FixationCross.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in FcComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Fc" ---
        for thisComponent in FcComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Fc.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'Try2'
    
    
    # --- Prepare to start Routine "Finish" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Finish.started', globalClock.getTime())
    Resp3Space.keys = []
    Resp3Space.rt = []
    _Resp3Space_allKeys = []
    # keep track of which components have finished
    FinishComponents = [back7, TnxParticipants, Resp3Space]
    for thisComponent in FinishComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Finish" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *back7* updates
        
        # if back7 is starting this frame...
        if back7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            back7.frameNStart = frameN  # exact frame index
            back7.tStart = t  # local t and not account for scr refresh
            back7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(back7, 'tStartRefresh')  # time at next scr refresh
            # update status
            back7.status = STARTED
            back7.setAutoDraw(True)
        
        # if back7 is active this frame...
        if back7.status == STARTED:
            # update params
            pass
        
        # *TnxParticipants* updates
        
        # if TnxParticipants is starting this frame...
        if TnxParticipants.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            TnxParticipants.frameNStart = frameN  # exact frame index
            TnxParticipants.tStart = t  # local t and not account for scr refresh
            TnxParticipants.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(TnxParticipants, 'tStartRefresh')  # time at next scr refresh
            # update status
            TnxParticipants.status = STARTED
            TnxParticipants.setAutoDraw(True)
        
        # if TnxParticipants is active this frame...
        if TnxParticipants.status == STARTED:
            # update params
            pass
        
        # *Resp3Space* updates
        waitOnFlip = False
        
        # if Resp3Space is starting this frame...
        if Resp3Space.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Resp3Space.frameNStart = frameN  # exact frame index
            Resp3Space.tStart = t  # local t and not account for scr refresh
            Resp3Space.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Resp3Space, 'tStartRefresh')  # time at next scr refresh
            # update status
            Resp3Space.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Resp3Space.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Resp3Space.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Resp3Space.status == STARTED and not waitOnFlip:
            theseKeys = Resp3Space.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Resp3Space_allKeys.extend(theseKeys)
            if len(_Resp3Space_allKeys):
                Resp3Space.keys = _Resp3Space_allKeys[-1].name  # just the last key pressed
                Resp3Space.rt = _Resp3Space_allKeys[-1].rt
                Resp3Space.duration = _Resp3Space_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in FinishComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Finish" ---
    for thisComponent in FinishComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Finish.stopped', globalClock.getTime())
    # the Routine "Finish" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
