# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:48:47 2017

@author: mkeranen

This script analyzes and processes the reflected light intensity signal 
obtained using an LED and blood.

TODO: Modularize and clean up code...

"""

import matplotlib.pyplot as plt
from openpyxl import load_workbook
import numpy as np
import scipy.signal as signal

#Load raw data from worksheet
fileName = 'sat_signal_data.xlsx'
wb = load_workbook(fileName, data_only=True) #data_only to pick data not formulas
sheet1 = wb.get_sheet_by_name('Raw_Data')
data = sheet1['A2' : 'A40101']

#Extract values from cells and store in list
dataList = []
for cell in data:
    dataList.append(cell[0].value)

#frequency in samples/second
samplingFrequency = 10000
timeStep = 1.0 / samplingFrequency
numSamples = len(dataList)

#Segment out portions of signal, eliminate first and last percentile to 
#disregard signal values during switch between signal segments
segment1 = dataList[int(numSamples * 0.01) : int(numSamples * 0.24)]
segment2 = dataList[int(numSamples * 0.26) : int(numSamples * 0.49)]
segment3 = dataList[int(numSamples * 0.51) : int(numSamples * 0.74)]
segment4 = dataList[int(numSamples * 0.76) : int(numSamples * 0.99)]

#Make segments length a power of 2 for DFT / FFT algorithm
segment1 = segment1[:-(len(segment1)-8192)]
segment2 = segment2[:-(len(segment2)-8192)]
segment3 = segment3[:-(len(segment3)-8192)]
segment4 = segment4[:-(len(segment4)-8192)]

#Create time step list
timeSegment = np.arange(0,len(segment1)/10000,timeStep)

#Discrete Fourier Transform for all segments
s1FFT = np.fft.fft(segment1)
s1FFTmag = np.absolute(s1FFT)
s1FFTfreq = np.fft.fftfreq(len(segment1), d=timeStep)

s2FFT = np.fft.fft(segment2)
s2FFTmag = np.absolute(s2FFT)
s2FFTfreq = np.fft.fftfreq(len(segment2), d=timeStep)

s3FFT = np.fft.fft(segment3)
s3FFTmag = np.absolute(s3FFT)
s3FFTfreq = np.fft.fftfreq(len(segment3), d=timeStep)

s4FFT = np.fft.fft(segment4)
s4FFTmag = np.absolute(s4FFT)
s4FFTfreq = np.fft.fftfreq(len(segment4), d=timeStep)

#Second order butterworth filter
N = 2       
cutoffFrequency = 100 #Hz
Wn = (2/samplingFrequency)*cutoffFrequency
b, a = signal.butter(N, Wn)

#Compute the frequency response of a digital filter
w, h = signal.freqz(b, a)

#Convert units from radians/sample to Hz
w = (w * samplingFrequency)/(2*3.14159)

#Implementation of butterworth filter
filteredSegment1 = signal.filtfilt(b, a, segment1)
filteredSegment2 = signal.filtfilt(b, a, segment2)
filteredSegment3 = signal.filtfilt(b, a, segment3)
filteredSegment4 = signal.filtfilt(b, a, segment4)

#Figures-----------------------------------------------------------------------
#Raw Signals
fig, ax = plt.subplots(4,1, sharex=True)
fig.suptitle('Raw Sat Signals', fontweight='bold')
plt.xlabel('Time [sec]')

ax[0].plot(timeSegment, segment1, 'b-', linewidth = 0.3)
ax[0].set_title('Segment 1')

ax[1].plot(timeSegment, segment2, 'b-', linewidth = 0.3)
ax[1].set_title('Segment 2')

ax[2].plot(timeSegment, segment3, 'b-', linewidth = 0.3)
ax[2].set_title('Segment 3')

ax[3].plot(timeSegment, segment4, 'b-', linewidth = 0.3)
ax[3].set_title('Segment 4')

plt.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('Raw Sat Signals.pdf', bbox_inches='tight')

#FFT results: Plot FFT Frequency vs Magnitude, up to Nyquist Frequency (Fs/2)
fig, ax = plt.subplots(4,1, sharex=True)
fig.suptitle('Sat Signal FFT', fontweight='bold')
plt.xlabel('Frequency [Hz]')

ax[0].plot(s1FFTfreq[1:int(len(s1FFTfreq)/2)], s1FFTmag[1:int(len(s1FFTfreq)/2)], 'b-', linewidth = 0.5)
ax[0].plot([cutoffFrequency,cutoffFrequency],[0,max(s1FFTmag[1:int(len(s1FFTfreq)/2)])],'y-', linewidth = 1)
ax[0].set_title('FFT Segment 1')

ax[1].plot(s1FFTfreq[1:int(len(s2FFTfreq)/2)], s2FFTmag[1:int(len(s2FFTfreq)/2)], 'g-', linewidth = 0.5)
ax[1].plot([cutoffFrequency,cutoffFrequency],[0,max(s2FFTmag[1:int(len(s2FFTfreq)/2)])],'y-', linewidth = 1)
ax[1].set_title('FFT Segment 2')

ax[2].plot(s1FFTfreq[1:int(len(s3FFTfreq)/2)], s3FFTmag[1:int(len(s3FFTfreq)/2)], 'r-', linewidth = 0.5)
ax[2].plot([cutoffFrequency,cutoffFrequency],[0,max(s3FFTmag[1:int(len(s3FFTfreq)/2)])],'y-', linewidth = 1)
ax[2].set_title('FFT Segment 3')

ax[3].plot(s1FFTfreq[1:int(len(s4FFTfreq)/2)], s4FFTmag[1:int(len(s4FFTfreq)/2)], 'k-', linewidth = 0.5)
ax[3].plot([cutoffFrequency,cutoffFrequency],[0,max(s4FFTmag[1:int(len(s4FFTfreq)/2)])],'y-', linewidth = 1)
ax[3].set_title('FFT Segment 4')

plt.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('Sat Signal FFT.pdf', bbox_inches='tight')

#Plot Filter Frequency Response
fig, ax = plt.subplots(1,1)
ax.plot(w, 20 * np.log10(abs(h)),'b-')
plt.xscale('log')
plt.title('Butterworth Filter Frequency Response', fontweight='bold')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(cutoffFrequency, color='green') # cutoff frequency
plt.ylim(-80,5)
plt.tight_layout()
plt.savefig('Filter Frequency Response.pdf', bbox_inches='tight')

#Filtered signals overlaid on raw signals
fig, ax = plt.subplots(4,1, sharex=True)
fig.suptitle('Filtered Sat Signals', fontweight='bold')
plt.xlabel('Time [sec]')

ax[0].plot(timeSegment, segment1, 'b-', linewidth = 0.3)
ax[0].plot(timeSegment, filteredSegment1, 'r-', linewidth = 1)
ax[0].set_title('Segment 1 w/ Filter')


ax[1].plot(timeSegment, segment2, 'b-', linewidth = 0.3)
ax[1].plot(timeSegment, filteredSegment2, 'r-', linewidth = 1)
ax[1].set_title('Segment 2 w/ Filter')

ax[2].plot(timeSegment, segment3, 'b-', linewidth = 0.3)
ax[2].plot(timeSegment, filteredSegment3, 'r-', linewidth = 1)
ax[2].set_title('Segment 3 w/ Filter')

ax[3].plot(timeSegment, segment4, 'b-', linewidth = 0.3)
ax[3].plot(timeSegment, filteredSegment4, 'r-', linewidth = 1)
ax[3].set_title('Segment 4 w/ Filter')

plt.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('Sat Signals with overlaid Filtered Signals.pdf', bbox_inches='tight')

plt.show()
