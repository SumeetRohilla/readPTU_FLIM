# readPTU_FLIM Library
### For demo: Jupyter Notebook to work with PicoQuant PTU files
**`The library provides the capability to handle all PicoQuant's TCSPC Harps in T2 as well as T3 mode.`**
- PicoQuant uses a bespoke file format called PTU to store data from time-tag-time-resolved (TTTR) measurements.<br/>
- Current file format (.ptu) and is subsequent to the former .pt2 or .pt3 and can handle both T2 and T3 acquisition modes for a variety of TCSPC devices (MultiHarp, HydraHarp, TimeHarp, PicoHarp, etc.) <br/>
- At the moment, the library was tested for FLIM data obtained using MultiHarp, HydraHarp and PicoHarp. <br/>

### **`What is not available !`**
- Option to save data after processing. <br/>
#### !!! Help is only an email away for MATLAB implementation of the same library.

### Dependencies

`readPTU_FLIM.py, numpy, matplotlib, numba,`

```python
from readPTU_FLIM import PTUreader 
import numpy as np from matplotlib 
import pyplot as plt
```

## Use case
- As an example, first we import the readPTU_FLIM library. Then we should open the PTU file using a PTUreader() object. By constructing a PTUreader() object, automatically opens the file, reads the PTU file header and TTTR data in raw format. PTU file header contains important measurement information (PQ TCSPC unit, laser/pulse sync rate, FLIM record type, scanner type, etc.)<br/><br/> 
- **Test_FLIM_image_daisyPollen_PicoHarp_2.ptu**<br> 
[Download the test file here](https://drive.google.com/file/d/1XtGL2yh_hJhaXIJhEDD5BpHNQXYQZX_p/view?usp=sharing)
FLIM data was acquired using a PicoHarp (T3 mode) <br>
Exctiation Laser light: 485 nm<br>
Number of Detection Channels: 2<br/><br/>
- At the moment, PTUreader library contains implementation for reading imaging data from scanner types: 
  - laser beam scanner (LSM) 
  - Piezo scanner

## Select a PTU file
```python
ptu_file  = PTUreader('Test_FLIM_image_daisyPollen_PicoHarp_2.ptu', print_header_data = False)
```

## How to access raw data from ptu_file object?
```python
# # Raw data can be accessed using ptu_file object
# sync    = ptu_file.sync    # Macro photon arrival time
# tcspc   = ptu_file.tcspc   # Micro photon arrival time (tcspc time bin resoultion)
# channel = ptu_file.channel # Detection channel of tcspc unit (<=8 for PQ hardware in 2019)
# special = ptu_file.special # Special event markers, for e.g. Frame, LineStart, LineStop, etc.
```

## Get flim_data_stack and intensity_image from raw TTTR data
 **Note**: once NEXT CODE block is executed raw TTTR data variables (sync, tcspc, channel, special) are deleted
- flim_data_stack: (pixX, pixY, spectral_detection_channel, tcspc_bins)
- intensity_image: (pixX, pixY)

**For example**:
How to get all the tcspc histogram data from all pixels from spectral detection channel 1?  (hint below!) <br /> 

```python
flim_data_stack, intensity_image = ptu_file.get_flim_data_stack()
# channel_1_data = flim_data_stack[:,:,0,:]
```

## Plot Intensity image
```
# %matplotlib inline
# #plot intensity image
# plt.imshow(intensity_image)
# plt.colorbar()
```

## Plot & Play ▶
- Please see Jupyter notebook for implementation
- Once the FLIM data stack is loaded, one could create custom gui for performing all sort of FLIM analysis routines as needed.
- As an example a simple intensity image is plotted 
- User can investigate the fluorescence decays after drawing a rectangle on the intensity image <br/>
![Interactive Demo Snapshot](Test_FLIM_image_daisyPollen_PicoHarp_2.png)

## Updates
- 27 Aug, 2019: Piezo Scanner data readability added to the library
