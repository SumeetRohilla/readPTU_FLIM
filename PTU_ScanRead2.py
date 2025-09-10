# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:11:52 2024

@author: narai
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:46:33 2024

@author: narai
"""
import struct
# import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pickle
from tqdm import tqdm
# from fast_histogram import histogramdd  
# from mpi4py import MPI 
from multiprocessing import Pool, cpu_count
# from numba import njit 

#%%


def cwaitbar(progress, total, message="Progress"):
    bar_length = 40  # Length of the progress bar in characters
    fraction = progress / total
    arrow = '-' * int(round(fraction * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r{message}: [{arrow}{spaces}] {int(fraction * 100)}%')
    sys.stdout.flush()

    if progress == total:
        sys.stdout.write('\n')
        
class PTUreader():
    
    """
    PTUreader() provides the capability to retrieve raw_data or image_data from a PTU file acquired using available PQ TCSPC module in the year 2019
    
    Input arguements:
    
    filename= path + filename
    print_header  = True or False
    
    Output
    
    ptu_read_raw_data     = This function reads single-photon data from the file 'name'
                            The output variables contain the followig data:
                            sync       : number of the sync events that preceeded this detection event
                            tcspc      : number of the tcspc-bin of the event
                            channel    : number of the input channel of the event (detector-number)
                            special    : marker event-type (0: photon; else : virtual photon/line_Startmarker/line_Stopmarker/framer_marker)
                            
    For example: Please see Jupyter notebook for additional info on how to get raw TTTR data
                    
    get_flim_data_stack    = This function builds a FLIM image from raw tttr_data
                             Outputs: flim_data_stack  = [numPixX numPixY numDetectors numTCSPCbins]
    
    """
    
    # Global constants
    # Define different tag types in header
    
    tag_type = dict(
    tyEmpty8      = 0xFFFF0008,
    tyBool8       = 0x00000008,
    tyInt8        = 0x10000008,
    tyBitSet64    = 0x11000008,
    tyColor8      = 0x12000008,
    tyFloat8      = 0x20000008,
    tyTDateTime   = 0x21000008,
    tyFloat8Array = 0x2001FFFF,
    tyAnsiString  = 0x4001FFFF,
    tyWideString  = 0x4002FFFF,
    tyBinaryBlob  = 0xFFFFFFFF,
    )
    
    # Dictionary with Record Types format for different TCSPC devices and corresponding T2 or T3 TTTR mode
    rec_type = dict(
        rtPicoHarpT3     = 0x00010303,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $03 (PicoHarp)
        rtPicoHarpT2     = 0x00010203,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $03 (PicoHarp)
        rtHydraHarpT3    = 0x00010304,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarpT2    = 0x00010204,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtHydraHarp2T3   = 0x01010304,  # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $03 (T3), HW: $04 (HydraHarp)
        rtHydraHarp2T2   = 0x01010204,  # (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $02 (T2), HW: $04 (HydraHarp)
        rtTimeHarp260NT3 = 0x00010305,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $05 (TimeHarp260N)
        rtTimeHarp260NT2 = 0x00010205,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $05 (TimeHarp260N)
        rtTimeHarp260PT3 = 0x00010306,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $06 (TimeHarp260P)
        rtTimeHarp260PT2 = 0x00010206,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $06 (TimeHarp260P)
        rtMultiHarpNT3   = 0x00010307,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $07 (MultiHarp150N)
        rtMultiHarpNT2   = 0x00010207,  # (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $07 (MultiHarp150N)
    )

    def __init__(self, filename, print_header_data = False):
        
        # raw_tttr_data = False, get_image_data = True
        # if get_image_data = True then get_raw_data = False
        # else get_raw_data = True and get_image_data = False
        # Usually a user will only demand for raw or image data
                
        #Reverse mappins of tag-type and record-type dictionary
        self.tag_type_r = {j: k for k, j in self.tag_type.items()}
        self.rec_type_r = {j: k for k, j in self.rec_type.items()}
        
        self.ptu_name        = filename
        self.print_header    = print_header_data
        
        f = open(self.ptu_name, 'rb')
        self.ptu_data_string = f.read() # ptu_data_string is a string of bytes and reads all file in memory
        f.close()

        
        #Check if the input file is a valid input file
        # Read magic and version of the PTU File
        self.magic = self.ptu_data_string[:8].rstrip(b'\0')
        self.version = self.ptu_data_string[8:16].rstrip(b'\0')
        if self.magic != b'PQTTTR':
            raise IOError("This file is not a valid PTU file. ")
            exit(0)
                

        
        self.head        = {}
        
        # Read  and print header if set True
        self._ptu_read_head(self.ptu_data_string)
        
        # # Read and return Raw TTTR Data
        # self._ptu_read_raw_data()
        self.sync = None
        self.tcspc = None
        self.channel = None
        self.special = None
        self.num     = None
        self.loc     = None
        self.num_records = self.head['TTResult_NumberOfRecords']
        
        if self.print_header == True:
            return self._print_ptu_head()
        
        return None
    
    def _ptu_TDateTime_to_time(self, TDateTime):

        EpochDiff = 25569  # days between 30/12/1899 and 01/01/1970
        SecsInDay = 86400  # number of seconds in a day

        return (TDateTime - EpochDiff) * SecsInDay

    
    def _ptu_read_tags(self, ptu_data_string, offset):

        # Get the header struct as a tuple
        # Struct fields: 32-char string, int32, uint32, int64

        tag_struct = struct.unpack('32s i I q', ptu_data_string[offset:offset+48])
        offset += 48

        # Get the tag name (first element of the tag_struct)
        tagName = tag_struct[0].rstrip(b'\0').decode()

        keys = ('idx', 'type', 'value')
        tag = {k: v for k, v in zip(keys, tag_struct[1:])}

        # Recover the name of the type from tag_dictionary
        tag['type'] = self.tag_type_r[tag['type']]
        tagStringR='NA'

        # Some tag types need conversion to appropriate data format
        if tag['type'] == 'tyFloat8':
            tag['value'] = np.int64(tag['value']).view('float64')
        elif tag['type'] == 'tyBool8':
            tag['value'] = bool(tag['value'])
        elif tag['type'] == 'tyTDateTime':
            TDateTime = np.uint64(tag['value']).view('float64')
            t = time.gmtime(self._ptu_TDateTime_to_time(TDateTime))
            tag['value'] = time.strftime("%Y-%m-%d %H:%M:%S", t)

        # Some tag types have additional data
        if tag['type'] == 'tyAnsiString':
            try: tag['data'] = ptu_data_string[offset: offset + tag['value']].rstrip(b'\0').decode()
            except: tag['data'] = ptu_data_string[offset: offset + tag['value']].rstrip(b'\0').decode(encoding  = 'utf-8', errors = 'ignore')
            tagStringR=tag['data']
            offset += tag['value']
        elif tag['type'] == 'tyFloat8Array':
            tag['data'] = np.frombuffer(ptu_data_string, dtype='float', count=tag['value']/8)
            offset += tag['value']
        elif tag['type'] == 'tyWideString':
            # WideString default encoding is UTF-16.
            tag['data'] = ptu_data_string[offset: offset + tag['value']*2].decode('utf16')
            tagStringR=tag['data']
            offset += tag['value']
        elif tag['type'] == 'tyBinaryBlob':
            tag['data'] = ptu_data_string[offset: offset + tag['value']]
            offset += tag['value']

        tagValue  = tag['value']

        return tagName, tagValue, offset, tagStringR
    
    
    def _ptu_read_head(self, ptu_data_string):
    
        offset         = 16
        FileTagEnd     = 'Header_End' 
        tag_end_offset = ptu_data_string.find(FileTagEnd.encode())

        tagName, tagValue, offset, tagString  = self._ptu_read_tags(ptu_data_string, offset)
        self.head[tagName] = tagValue

        #while offset < tag_end_offset:
        while tagName != FileTagEnd:
                tagName, tagValue, offset, tagString = self._ptu_read_tags(ptu_data_string, offset)
                if tagString=='NA': self.head[tagName] = tagValue
                else: self.head[tagName] = tagString
#                 print(tagName, tagValue)

        # End of Header file and beginning of TTTR data
        self.head[FileTagEnd] = offset


    def _print_ptu_head(self): 
        #print "head" dictionary     
        print("{:<30} {:8}".format('Head ID','Value'))

        for keys in self.head:
            val = self.head[keys] 
            print("{:<30} {:<8}".format(keys, val))     
    
    def _ptu_read_raw_data(self, cnts = None, head = None):
    
        '''
        This function reads single-photon data from the file 's'

        Returns:
        sync    : number of the sync events that preceeded this detection event
        tcspc   : number of the tcspc-bin of the event
        chan    : number of the input channel of the event (detector-number)
        special : indicator of the event-type (0: photon; else : virtual photon)
        num     : counter of the records that were actually read


        '''

        if cnts is None or len(cnts) == 0:
            cnts = [0, 0]
        elif len(cnts) < 2:
            cnts = [0, cnts[0]]

        if cnts[1] > 0:
            with open(self.ptu_name, 'rb') as fid:
                fid.seek(self.head['Header_End'], 0)

                if cnts[0] > 1:
                    fid.seek(4 * (cnts[0] - 1), 1)
                
                t3records = np.fromfile(fid, dtype='uint32', count=cnts[1])
                num = t3records.size
        else:
            t3records = np.frombuffer(self.ptu_data_string, dtype='uint32', offset=self.head['Header_End'])
            num = t3records.size
            
        record_type = self.rec_type_r[self.head['TTResultFormat_TTTRRecType']]
        # num_T3records = self.head['TTResult_NumberOfRecords']
        
        print(record_type)
        #Next is to do T3Records formatting according to Record_type

        if record_type == 'rtPicoHarpT3':

            print('TCSPC Hardware: {}'.format(record_type[2:]))
            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | | | | | | | | | | |  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | |x|x|x|x|x|x|x|x|x|x|x|x|  | | | | | | | | | | | | | | | | |  --> TCSPC bin
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x|x|x|x| | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+

            WRAPAROUND = 65536                                                   # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 65535)                        # Lowest 16 bits
            tcspc      = np.bitwise_and(np.right_shift(t3records, 16), 4095)     # Next 12 bits, dtime can be obtained from header
            chan       = np.bitwise_and(np.right_shift(t3records, 28),15)        # Next 4 bits 
            special    = ((chan==15)*1)*(np.bitwise_and(tcspc,15)*1)               # Last bit for special markers
            
            index      = ((chan==15)*1)*((np.bitwise_and(tcspc,15)==0)*1)        # Find overflow locations
            chan[chan==15]  = special[chan==15]
            chan[chan==1] = 0
            chan[chan==2] = 1
            chan[chan==4] = 0
            
        elif record_type == 'rtPicoHarpT2':

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T2 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | |x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x|x|x|x| | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+

            WRAPAROUND = 210698240                                               # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 268435455)                    # Lowest 28 bits
            tcspc      = np.bitwise_and(t3records, 15)                           # Next 4 bits, dtime can be obtained from header
            chan       = np.bitwise_and(np.right_shift(t3records, 28),15)        # Next 4 bits 
            special    = ((chan==15)*1)*np.bitwise_and(tcspc,15)                 # Last bit for special markers
            index      = ((chan==15)*1)*((np.bitwise_and(tcspc,15)==0)*1)        # Find overflow locations

        elif record_type in ['rtHydraHarpT3', 'rtHydraHarp2T3', 'rtTimeHarp260NT3', 'rtTimeHarp260PT3','rtMultiHarpNT3']:

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | | | | | | | | | | |  | | | | | | |x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x| | | | | | | | | | |  --> TCSPC bin
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Special markers
            #   +-------------------------------+  +-------------------------------+
            WRAPAROUND = 1024                                                   # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 1023)                        # Lowest 10 bits
            tcspc      = np.bitwise_and(np.right_shift(t3records, 10), 32767)   # Next 15 bits, dtime can be obtained from header
            chan       = np.bitwise_and(np.right_shift(t3records, 25),63)       # Next 8 bits 
            special    = np.bitwise_and(t3records,2147483648)>0                 # Last bit for special markers
            index      = (special*1)*((chan==63)*1)                           # Find overflow locations
            special    = (special*1)*chan

        elif record_type == 'rtHydraHarpT2':

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Special markers
            #   +-------------------------------+  +-------------------------------+
            WRAPAROUND = 33552000                                               # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 33554431)                    # Lowest 25 bits
            chan       = np.bitwise_and(np.right_shift(t3records, 25),63)       # Next 6 bits 
            tcspc      = np.bitwise_and(chan, 15)                               
            special    = np.bitwise_and(np.right_shift(t3records, 31),1)        # Last bit for special markers
            index      = (special*1) * ((chan==63)*1)                             # Find overflow locations
            special    = (special*1)*chan

        elif record_type in ['rtHydraHarp2T2', 'rtTimeHarp260NT2', 'rtTimeHarp260PT2','rtMultiHarpNT2']:

            print('TCSPC Hardware: {}'.format(record_type[2:]))

            #   +----------------------+ T3 32 bit record  +---------------------+
            #   |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x| --> 32 bit record
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | | | | | | | |x|x|x|x|x|x|x|x|x|  |x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|x|  --> Sync
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   | |x|x|x|x|x|x| | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Spectral/TCSPC input Channel
            #   +-------------------------------+  +-------------------------------+
            #   +-------------------------------+  +-------------------------------+
            #   |x| | | | | | | | | | | | | | | |  | | | | | | | | | | | | | | | | |  --> Special markers
            #   +-------------------------------+  +-------------------------------+

            WRAPAROUND = 33554432                                               # After this sync counter will overflow
            sync       = np.bitwise_and(t3records, 33554431)                    # Lowest 25 bits
            chan       = np.bitwise_and(np.right_shift(t3records, 25),63)       # Next 6 bits 
            tcspc      = np.bitwise_and(chan, 15)                               
            special    = np.bitwise_and(np.right_shift(t3records, 31),1)        # Last bit for special markers
            index      = (special*1) * ((chan==63)*1)                            # Find overflow locations
            special    = (special*1)*chan

        else:
            print('Illegal RecordType!')
            exit(0)



        # Fill in the correct sync values for overflow location    
        #sync[np.where(index == 1)] = 1 # assert values of sync = 1 just right after overflow to avoid any overflow-correction instability in next step
        if record_type in ['rtHydraHarp2T3','rtTimeHarp260PT3','rtMultiHarpNT3']:
            sync    = sync + (WRAPAROUND*np.cumsum(index*sync)) # For HydraHarp V1 and TimeHarp260 V1 overflow corrections 
        else:
            sync    = sync + (WRAPAROUND*np.cumsum(index)) # correction for overflow to sync varibale

        sync     = np.delete(sync, np.where(index == 1), axis = 0)
        tcspc    = np.delete(tcspc, np.where(index == 1), axis = 0)
        chan     = np.delete(chan, np.where(index == 1), axis = 0)
        special  = np.delete(special, np.where(index == 1), axis = 0)
             
        # del index

        # Convert to appropriate data type to save memory

        self.sync    = sync.astype(np.uint64, copy=False)
        self.tcspc   = tcspc.astype(np.uint16, copy=False)
        self.channel = chan.astype(np.uint8,  copy=False)
        self.special = special.astype(np.uint8, copy=False)
        self.num     = num 
        last_zero_index = np.where(index == 0)[0][-1] if np.any(index == 0) else -1
        self.loc = num - (last_zero_index + 1) if last_zero_index != -1 else num
        

        # print("Raw Data has been Read!\n")

        return self.sync, self.tcspc, self.channel, self.special, self.num, self.loc
    
    
    
    
    def get_photon_chunk(self, start_idx, end_idx, head):
       """Get a chunk of photon data from start_idx to end_idx."""
       return self._ptu_read_raw_data([start_idx, end_idx], head)
    
def Process_Frame(im_sync,im_col,im_line,im_chan,im_tcspc,head,cnum = 1, resolution = 0.2):
    Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)  # resolution of 0.256 ns to calculate average lifetimes
    chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
    # SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
    nx = head['ImgHdr_PixX']
    ny = head['ImgHdr_PixY']
    dind = np.unique(im_chan).astype(np.int64)
    Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
    maxch_n = len(dind)

    tcspc_pix = np.zeros((nx, ny, Ngate, maxch_n*cnum), dtype=np.uint32) # X-Y-Tau-CH*Pulse
    # timeF = [None] * maxch_n*cnum
    tag = np.zeros((nx, ny, maxch_n, cnum), dtype=np.uint32) #XYCP
    tau = np.zeros((nx, ny, maxch_n, cnum))

    binT = np.transpose(np.tile(np.arange(Ngate).reshape(-1, 1, 1) * Resolution, (1, nx, ny)), (1, 2, 0))  # 3D time axis
    # the program currently divides the tcspc bins equally into cnum windows starting from 0th bin. 
    # A sophisticated program such as DetectorTimeGates can be implemented if needed
    
       
    
    # without mpi4py
    for ch in range(maxch_n):
        for p in range(cnum):
            ind = (im_chan == dind[ch]) & (im_tcspc<tmpCh/cnum*(p+1)) & (im_tcspc>=p*tmpCh/cnum)
            idx = np.where(ind)[0]
            # print(len(idx))
            tcspc_pix[:, :, :, ch*cnum + p] = mHist3(im_line[idx].astype(np.int64), 
                                        im_col[idx].astype(np.int64), 
                                        (im_tcspc[idx] / chDiv).astype(np.int64) - int(p*tmpCh/cnum/chDiv), 
                                        np.arange(nx), 
                                        np.arange(ny), 
                                        np.arange(Ngate))[0]  # tcspc histograms for all the pixels at once!
            
            # tcspc_pix[:, :, :, ch*cnum + p] = histogramdd((im_line[idx].astype(np.int64), 
            #                             im_col[idx].astype(np.int64), 
            #                             (im_tcspc[idx] / chDiv).astype(np.int64) - int(p*tmpCh/cnum/chDiv)), 
            #                             (nx, ny, Ngate),
            #                             [(0,nx-1),(0,ny-1),(0,Ngate-1)])  # tcspc histograms for all the pixels at once!
        
           
            tag[:, :, ch, p] = np.sum(tcspc_pix[:, :, :, ch*cnum + p], axis=2)
            tau[:, :, ch, p] = np.real(np.sqrt((np.sum(binT ** 2 * tcspc_pix[:, :, :, ch*cnum + p], axis=2) / (tag[:, :, ch, p] + 10**-10)) -
                                            (np.sum(binT * tcspc_pix[:, :, :, ch*cnum + p], axis=2) / (tag[:, :, ch, p] + 10**-10)) ** 2))
            # timeF[:,ch*cnum + p] = np.round(im_sync[idx] / SyncRate / Resolution / 1e-9) + im_tcspc[idx].astype(np.int64)  # in tcspc bins 
        
    return tag, tau, tcspc_pix


# def Process_FrameFast(im_sync,im_col,im_line,im_chan,im_tcspc,head,cnum = 1, resolution = 0.2):

#     Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)  # resolution of 0.256 ns to calculate average lifetimes
#     chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
#     # SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
#     nx = head['ImgHdr_PixX']
#     ny = head['ImgHdr_PixY']
#     dind = np.unique(im_chan).astype(np.int64)
#     Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
#     tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
#     maxch_n = len(dind)

#     # tcspc_pix = np.zeros((nx, ny, Ngate, maxch_n*cnum), dtype=np.uint32) # X-Y-Tau-CH*Pulse
#     # timeF = [None] * maxch_n*cnum
#     # tag = np.zeros((nx, ny, maxch_n, cnum), dtype=np.uint32) #XYCP
#     # tau = np.zeros((nx, ny, maxch_n, cnum))

#     # the program currently divides the tcspc bins equally into cnum windows starting from 0th bin. 
#     # A sophisticated program such as DetectorTimeGates can be implemented if needed
    
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()  # Get the rank of this process
#     sizeP = comm.Get_size()  # Get the total number of processes

#     # Assuming the following arrays/variables are already defined:
#     # im_chan, dind, im_tcspc, tmpCh, cnum, chDiv, im_line, im_col, tcspc_pix, nx, ny, Ngate, maxch_n
    
#     # Decide to split the workload across the nx dimension
#     chunk_size = nx // sizeP
#     start = rank * chunk_size
#     end = (rank + 1) * chunk_size if rank != sizeP - 1 else nx  # Ensure last process takes the remaining part
    
#     # Use im_col to filter only the relevant data for this process
#     idx_col = np.where((im_line >= start) & (im_line < end))[0]  # Indices relevant to this process
    
#     # Filter the other relevant arrays based on idx_col
#     im_line_chunk = im_line[idx_col]
#     im_col_chunk = im_col[idx_col]
#     im_tcspc_chunk = im_tcspc[idx_col]
#     im_chan_chunk = im_chan[idx_col]
#     im_tcspc_chunk = im_tcspc[idx_col]
    
#     # Initialize a zero array for this process's chunk of tcspc_pix
#     tcspc_pix_chunk = np.zeros((end - start, ny, Ngate, maxch_n * cnum), dtype=np.int64)
#     tag_chunk = np.zeros((end - start, ny, maxch_n, cnum), dtype=np.float64)
#     tau_chunk = np.zeros((end - start, ny, maxch_n, cnum), dtype=np.float64)
#     binT = np.transpose(np.tile(np.arange(Ngate).reshape(-1, 1, 1) * Resolution, (1, end - start, ny)), (1, 2, 0))  # 3D time axis
      
    
#     for ch in range(maxch_n):
#         for p in range(cnum):
#             ind = (im_chan_chunk == dind[ch]) & (im_tcspc_chunk<tmpCh/cnum*(p+1)) & (im_tcspc_chunk>=p*tmpCh/cnum)
#             idx = np.where(ind)[0]
#             # print(len(idx))
#             # Compute the histograms for the chunk of pixels this process is responsible for (i.e., along the nx dimension)
#             tcspc_pix_chunk[:, :, :, ch * cnum + p] = mHist3(
#                 im_line_chunk[idx].astype(np.int64),
#                 im_col_chunk[idx].astype(np.int64),
#                 (im_tcspc_chunk[idx] / chDiv).astype(np.int64) - int(p * tmpCh / cnum / chDiv),
#                 np.arange(start, end),  # The nx range handled by this process
#                 np.arange(ny),
#                 np.arange(Ngate)
#                 )[0]  # tcspc histograms for the current chunk of pixels
            
#             tag_chunk[:, :, ch, p] = np.sum(tcspc_pix_chunk[:, :, :, ch * cnum + p], axis=2)
#             tau_chunk[:, :, ch, p] = np.real(np.sqrt(
#                 (np.sum(binT ** 2 * tcspc_pix_chunk[:, :, :, ch * cnum + p], axis=2) / (tag_chunk[:, :, ch, p] + 1e-10)) -
#                 (np.sum(binT * tcspc_pix_chunk[:, :, :, ch * cnum + p], axis=2) / (tag_chunk[:, :, ch, p] + 1e-10)) ** 2
#                 ))
    
#     if rank == 0:
#         # Allocate the final arrays on the root process
#         tcspc_pix = np.zeros((nx, ny, Ngate, maxch_n * cnum), dtype=np.int64)
#         tag = np.zeros((nx, ny, maxch_n, cnum), dtype=np.float64)
#         tau = np.zeros((nx, ny, maxch_n, cnum), dtype=np.float64)
        
#     # Use MPI_Gather to gather all the chunks for tcspc_pix_chunk
#     comm.Gather(tcspc_pix_chunk, tcspc_pix if rank == 0 else None, root=0)

#     # Use MPI_Gather to gather all the chunks for tag_chunk and tau_chunk
#     comm.Gather(tag_chunk, tag if rank == 0 else None, root=0)
#     comm.Gather(tau_chunk, tau if rank == 0 else None, root=0)
    
#     if rank == 0:
#         # Now the root process (rank 0) has the complete tcspc_pix, tag, and tau arrays
#         print("Computation completed and gathered on rank 0.")
            
#         return tag, tau, tcspc_pix
#     else: 
#         return None, None, None

def Frame_update(tmptag, tmptau, tcspc, chan, line, col, head, cnum):
    Resolution = head['MeasDesc_Resolution'] * 1e9  # resolution of 0.256 ns to calculate average lifetimes
    dind = np.unique(chan).astype(np.int64)
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'])  # total number of channels in the original tcspc histogram
    maxch_n = len(dind)
    
    
    for ch in range(maxch_n):
        for p in range(cnum):
            for y in range(head['ImgHdr_PixY']):
                ind = (chan == dind[ch]) & \
                    (tcspc < tmpCh / cnum * (p + 1)) & \
                    (tcspc >= p * tmpCh / cnum) & \
                    (col == y)
                            
                idx = np.where(ind)[0]
                # print(len(idx))
                tmptag[line, y, ch, p] = len(idx)
                if len(idx)>2:
                    tmptau[line, y, ch, p] = np.sqrt(np.var(tcspc[idx].astype(np.int64) - int(p * tmpCh / cnum )))*Resolution
    return tmptag, tmptau        
            

    

def process_chunk(chunk_data):
    # Unpack the chunk data
    im_sync_chunk, im_col_chunk, im_line_chunk, im_chan_chunk,\
        im_tcspc_chunk, head, cnum, start, end, Resolution, chDiv,\
            dind, tmpCh, Ngate, maxch_n, ny = chunk_data
    
    # Initialize arrays for this chunk
    tcspc_pix_chunk = np.zeros((end - start, ny, Ngate, maxch_n * cnum), dtype=np.int64)
    tag_chunk = np.zeros((end - start, ny, maxch_n, cnum), dtype=np.float64)
    tau_chunk = np.zeros((end - start, ny, maxch_n, cnum), dtype=np.float64)
    binT = np.transpose(np.tile(np.arange(Ngate).reshape(-1, 1, 1) * Resolution, (1, end - start, ny)), (1, 2, 0))  # 3D time axis
    
    for ch in range(maxch_n):
        for p in range(cnum):
            ind = (im_chan_chunk == dind[ch]) & (im_tcspc_chunk < tmpCh / cnum * (p + 1)) & (im_tcspc_chunk >= p * tmpCh / cnum)
            idx = np.where(ind)[0]
            
            # Compute the histograms for the chunk of pixels
            tcspc_pix_chunk[:, :, :, ch * cnum + p] = mHist3(
                im_line_chunk[idx].astype(np.int64),
                im_col_chunk[idx].astype(np.int64),
                (im_tcspc_chunk[idx] / chDiv).astype(np.int64) - int(p * tmpCh / cnum / chDiv),
                np.arange(start, end),  # The nx range handled by this process
                np.arange(ny),
                np.arange(Ngate)
            )[0]  # tcspc histograms for the current chunk of pixels

            tag_chunk[:, :, ch, p] = np.sum(tcspc_pix_chunk[:, :, :, ch * cnum + p], axis=2)
            tau_chunk[:, :, ch, p] = np.real(np.sqrt(
                (np.sum(binT ** 2 * tcspc_pix_chunk[:, :, :, ch * cnum + p], axis=2) / (tag_chunk[:, :, ch, p] + 1e-10)) -
                (np.sum(binT * tcspc_pix_chunk[:, :, :, ch * cnum + p], axis=2) / (tag_chunk[:, :, ch, p] + 1e-10)) ** 2
            ))

    return tag_chunk, tau_chunk, tcspc_pix_chunk

def Process_FrameFast(im_sync, im_col, im_line, im_chan, im_tcspc, head, cnum=1, resolution=0.2):
    Resolution = max(head['MeasDesc_Resolution'] * 1e9, resolution)  # resolution of 0.256 ns to calculate average lifetimes
    chDiv = np.ceil(1e-9 * Resolution / head['MeasDesc_Resolution'])
    nx = head['ImgHdr_PixX']
    ny = head['ImgHdr_PixY']
    dind = np.unique(im_chan).astype(np.int64)
    Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
    tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'])  # total number of channels in the original tcspc histogram
    maxch_n = len(dind)

    # Number of processes to use (you can adjust this based on your CPU)
    num_processes = min(cpu_count(), 6)  # Use up to 4 processes or less depending on available cores

    # Split the nx dimension into chunks for each process
    chunk_size = nx // num_processes
    chunks = [(im_sync, im_col, im_line, im_chan, im_tcspc, head, cnum, i * chunk_size,
               (i + 1) * chunk_size if i != num_processes - 1 else nx, Resolution, chDiv, dind, tmpCh, Ngate, maxch_n, ny)
              for i in range(num_processes)]
    
    # Initialize the multiprocessing pool
    with Pool(processes=num_processes) as pool:
        # Distribute the work across processes
        results = pool.map(process_chunk, chunks)

    # Gather the results
    tcspc_pix = np.zeros((nx, ny, Ngate, maxch_n * cnum), dtype=np.int64)
    tag = np.zeros((nx, ny, maxch_n, cnum), dtype=np.float64)
    tau = np.zeros((nx, ny, maxch_n, cnum), dtype=np.float64)

    for i, (tag_chunk, tau_chunk, tcspc_pix_chunk) in enumerate(results):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_processes - 1 else nx
        tcspc_pix[start:end, :, :, :] = tcspc_pix_chunk
        tag[start:end, :, :, :] = tag_chunk
        tau[start:end, :, :, :] = tau_chunk

    return tag, tau, tcspc_pix

def mHist2(x, y, xv=None, yv=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    ind = ~np.isfinite(x) | ~np.isfinite(y)
    x = x[~ind]
    y = y[~ind]

    if xv is not None and yv is None:
        if len(xv) == 1:
            NX = xv
            NY = xv
        else:
            NX = xv[0]
            NY = xv[1]
    elif xv is None and yv is None:
         NX, NY = 100, 100
            
    if xv is not None and yv is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        dx, dy = (xmax - xmin) / NX, (ymax - ymin) / NY

        xv = np.linspace(xmin, xmax, NX)
        yv = np.linspace(ymin, ymax, NY)

        x = np.floor((x - xmin) / dx).astype(int)
        y = np.floor((y - ymin) / dy).astype(int)

        xmax = np.round((xmax-xmin)/dx).astype(int)
        ymax = np.round((ymax-ymin)/dy).astype(int)

    else:
        xmin, xmax = xv[0], xv[-1]
        ymin, ymax = yv[0], yv[-1]

        # clipping
        x = np.clip(x, xmin, xmax)
        y = np.clip(y, ymin, ymax)

        # Handling for x
        if (np.sum(np.diff(np.diff(xv))) == 0):
            dx = xv[1] - xv[0]
            x = np.int64(np.floor((x-xmin)/dx + 0.5))
            xmax =  np.int64(np.floor((xmax-xmin)/dx + 0.5))+1
        else:
            x = np.round(np.interp(x, xv, np.arange(len(xv)))).astype(int)
            xmax = np.round(np.interp(xmax, xv, np.arange(len(xv)))).astype(int)
            
        # Handling for y
        if np.sum(np.diff(np.diff(yv))) == 0:
            dy = yv[1] - yv[0]
            y = np.int64(np.floor((y-ymin)/dy + 0.5))
            ymax =  np.int64(np.floor((ymax-ymin)/dy + 0.5))+1
        else:
            y = np.round(np.interp(y, yv, np.arange(len(yv)))).astype(int)
            ymax = np.round(np.interp(ymax, yv, np.arange(len(yv)))).astype(int)

    # Initialize the histogram array
    h = np.zeros(len(xv) * len(yv), dtype=int)
    num = np.sort(x + xmax * y)
    np.add.at(h, num, 1)
    
    tmp = np.diff((np.diff(np.concatenate(([-1], num, [-1]))) == 0).astype(int))

    ind = np.arange(len(num))
    # Calculate the flattened indices for the histogram
    h[num[tmp == 1]] += -ind[tmp == 1] + ind[tmp == -1]
    
    # Increment the histogram at the calculated indices
    h = h.reshape((len(xv), len(yv)), order='F')

    return h, xv, yv


# @njit
def mHist3(x, y, z, xv=None, yv=None, zv=None):
    # x = np.asarray(x).flatten()
    # y = np.asarray(y).flatten()
    # z = np.asarray(z).flatten()
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    
    ind = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(z)
    x = x[~ind]
    y = y[~ind]
    z = z[~ind]

    if xv is not None and yv is None and zv is None:
        if len(xv)==1:
            NX = xv
            NY = xv
            NZ = xv
        else:
            NX = xv[0]
            NY = xv[1]
            NZ = xv[2]
    elif xv is None and yv is None and zv is None:
         NX, NY, NZ = 100, 100, 100
            
    if xv is not None and yv is not None and zv is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)
        dx, dy, dz = (xmax - xmin) / NX, (ymax - ymin) / NY, (zmax - zmin) / NZ

        xv = np.linspace(xmin, xmax, NX)
        yv = np.linspace(ymin, ymax, NY)
        zv = np.linspace(zmin, zmax, NZ)

        x = np.floor((x - xmin) / dx).astype(int)
        y = np.floor((y - ymin) / dy).astype(int)
        z = np.floor((z - zmin) / dz).astype(int)

        xmax = np.round((xmax-xmin)/dx).astype(int)
        ymax = np.round((ymax-ymin)/dx).astype(int)

    else:
        xmin, xmax = xv[0], xv[-1]
        ymin, ymax = yv[0], yv[-1]
        zmin, zmax = zv[0], zv[-1]

        # clipping
        x = np.clip(x, xmin, xmax, out = x, casting='unsafe')
        y = np.clip(y, ymin, ymax, out = y, casting='unsafe')
        z = np.clip(z, zmin, zmax, out = z, casting='unsafe')

        # Handling for x
        if (np.sum(np.diff(np.diff(xv))) == 0):
            dx = xv[1] - xv[0]
            # x = np.round((x - xmin) / dx).astype(int)
            x = np.int64(np.floor((x-xmin)/dx + 0.5))
            xmax =  np.int64(np.floor((xmax-xmin)/dx + 0.5))+1
        else:
            x = np.round(np.interp(x, xv, np.arange(len(xv)))).astype(int)
            xmax = np.round(np.interp(xmax, xv, np.arange(len(xv)))).astype(int)
            
        # Handling for y
        if np.sum(np.diff(np.diff(yv))) == 0:
            dy = yv[1] - yv[0]
            # y = np.round((y - ymin) / dy).astype(int)
            y = np.int64(np.floor((y-ymin)/dy + 0.5))
            ymax =  np.int64(np.floor((ymax-ymin)/dy + 0.5))+1
        else:
            y = np.round(np.interp(y, yv, np.arange(len(yv)))).astype(int)
            ymax = np.round(np.interp(ymax, yv, np.arange(len(yv)))).astype(int)
            
        # Handling for z
        if np.sum(np.diff(np.diff(zv))) == 0:
            dz = zv[1] - zv[0]
            # z = np.round((z - zmin) / dz).astype(int)
            z = np.int64(np.floor((z-zmin)/dz + 0.5))
            zmax =  np.int64(np.floor((zmax-zmin)/dz + 0.5))+1
        else:
            z = np.round(np.interp(z, zv, np.arange(len(zv)))).astype(int)
            zmax = np.round(np.interp(zmax, zv, np.arange(len(zv)))).astype(int)


    # Initialize the histogram array
    h = np.zeros(len(xv)* len(yv)* len(zv), dtype=int)
    # h = np.zeros((len(xv), len(yv), len(zv)), dtype=int)
    
    num = np.sort(x + xmax * y + xmax * ymax * z)
    np.add.at(h, num, 1)
    # np.add.at(h.ravel(), num, 1)
    # h[num]=1
    
    tmp = np.diff((np.diff(np.concatenate(([-1],num,[-1])))==0).astype(int))
    # tmp = np.diff(np.diff(np.concatenate(([-1], num, [-1]))) == 0).astype(int)

    ind = np.arange(len(num))
    # Calculate the flattened indices for the histogram
    h[num[tmp==1]] += -ind[tmp==1] + ind[tmp==-1]
    

    # Increment the histogram at the calculated indices
    
    h = h.reshape((len(xv), len(yv), len(zv)), order='F')
    # h.ravel()[num[tmp == 1]] += -ind[tmp == 1] + ind[tmp == -1]

    return h, xv, yv, zv


# @njit
def mHist4(x, y, z, t, xv=None, yv=None, zv=None, tv=None):
    # Convert inputs to flattened arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    t = np.asarray(t).ravel()
    
    # Remove non-finite values
    ind = ~np.isfinite(x) | ~np.isfinite(y) | ~np.isfinite(z) | ~np.isfinite(t)
    x = x[~ind]
    y = y[~ind]
    z = z[~ind]
    t = t[~ind]

    # Set default grid sizes
    if xv is None and yv is None and zv is None and tv is None:
        NX, NY, NZ, NT = 100, 100, 100, 100
    elif xv is not None and yv is None and zv is None and tv is None:
        if len(xv) == 1:
            NX = NY = NZ = NT = xv
        else:
            NX, NY, NZ, NT = xv[0], xv[1], xv[2], xv[3]
    elif xv is not None and yv is not None and zv is not None and tv is None:
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)
        tmin, tmax = np.min(t), np.max(t)
        dx, dy, dz, dt = (xmax - xmin) / NX, (ymax - ymin) / NY, (zmax - zmin) / NZ, (tmax - tmin) / NT

        xv = np.linspace(xmin, xmax, NX)
        yv = np.linspace(ymin, ymax, NY)
        zv = np.linspace(zmin, zmax, NZ)
        tv = np.linspace(tmin, tmax, NT)

        x = np.floor((x - xmin) / dx).astype(int)
        y = np.floor((y - ymin) / dy).astype(int)
        z = np.floor((z - zmin) / dz).astype(int)
        t = np.floor((t - tmin) / dt).astype(int)

        xmax = np.round((xmax - xmin) / dx).astype(int)
        ymax = np.round((ymax - ymin) / dy).astype(int)
        zmax = np.round((zmax - zmin) / dz).astype(int)
        tmax = np.round((tmax - tmin) / dt).astype(int)

    else:
        xmin, xmax = xv[0], xv[-1]
        ymin, ymax = yv[0], yv[-1]
        zmin, zmax = zv[0], zv[-1]
        tmin, tmax = tv[0], tv[-1]

        # Clipping values to ensure they are within the bounds
        x = np.clip(x, xmin, xmax)
        y = np.clip(y, ymin, ymax)
        z = np.clip(z, zmin, zmax)
        t = np.clip(t, tmin, tmax)

        # Handling for x
        if np.sum(np.diff(np.diff(xv))) == 0:
            dx = xv[1] - xv[0]
            x = np.int64(np.floor((x - xmin) / dx + 0.5))
            xmax = np.int64(np.floor((xmax - xmin) / dx + 0.5)) + 1
        else:
            x = np.round(np.interp(x, xv, np.arange(len(xv)))).astype(int)
            xmax = np.round(np.interp(xmax, xv, np.arange(len(xv)))).astype(int)
        
        # Handling for y
        if np.sum(np.diff(np.diff(yv))) == 0:
            dy = yv[1] - yv[0]
            y = np.int64(np.floor((y - ymin) / dy + 0.5))
            ymax = np.int64(np.floor((ymax - ymin) / dy + 0.5)) + 1
        else:
            y = np.round(np.interp(y, yv, np.arange(len(yv)))).astype(int)
            ymax = np.round(np.interp(ymax, yv, np.arange(len(yv)))).astype(int)
        
        # Handling for z
        if np.sum(np.diff(np.diff(zv))) == 0:
            dz = zv[1] - zv[0]
            z = np.int64(np.floor((z - zmin) / dz + 0.5))
            zmax = np.int64(np.floor((zmax - zmin) / dz + 0.5)) + 1
        else:
            z = np.round(np.interp(z, zv, np.arange(len(zv)))).astype(int)
            zmax = np.round(np.interp(zmax, zv, np.arange(len(zv)))).astype(int)
        
        # Handling for t
        if np.sum(np.diff(np.diff(tv))) == 0:
            dt = tv[1] - tv[0]
            t = np.int64(np.floor((t - tmin) / dt + 0.5))
            tmax = np.int64(np.floor((tmax - tmin) / dt + 0.5)) + 1
        else:
            t = np.round(np.interp(t, tv, np.arange(len(tv)))).astype(int)
            tmax = np.round(np.interp(tmax, tv, np.arange(len(tv)))).astype(int)

    # Initialize the histogram array
    h = np.zeros(len(xv) * len(yv) * len(zv) * len(tv), dtype=int)
    num = np.sort(x + xmax * y + xmax * ymax * z + xmax * ymax * zmax * t)
    np.add.at(h, num, 1)
    
    tmp = np.diff((np.diff(np.concatenate(([-1], num, [-1]))) == 0).astype(int))

    ind = np.arange(len(num))
    h[num[tmp == 1]] += -ind[tmp == 1] + ind[tmp == -1]
    
    # Reshape the histogram to 4D
    h = h.reshape((len(xv), len(yv), len(zv), len(tv)), order='F')

    return h, xv, yv, zv, tv


   
def PTU_ScanRead(filename, cnum = 1, plt_flag=False):
    # cnum is an additional input to separate the photons while calculating 
    photons = int(1e7) # number of photons to read at a time. Can be adjusted based on the system memory
    
    ptu_reader = PTUreader(filename)
    head = ptu_reader.head
    
    if not head:
        print("Header data could not be read. Aborting...")
        return None, None, None, None, None, None, None

    nx = head['ImgHdr_PixX']
    ny = head['ImgHdr_PixY']

    if head['ImgHdr_Ident'] in [1, 6]:  # Scan Cases 1 and 6
      
        # Common settings
        anzch = 32  # max number of channels (can be 64 now for the new MultiHarp)
        # Resolution = max([1e9 * head['MeasDesc_Resolution'], 0.064])
        Resolution = 1e9*head['MeasDesc_Resolution']
        chDiv = 1e-9 * Resolution / head['MeasDesc_Resolution']
        Ngate = int(np.ceil(1e9 * head['MeasDesc_GlobalResolution'] / Resolution)) + 1
        head['MeasDesc_Resolution'] = Resolution * 1e-9

        # defaults
        LineStart = 4
        LineStop = 2
        
        if 'ImgHdr_LineStart' in head:
            LineStart = 2 ** (head['ImgHdr_LineStart'] - 1)
        if 'ImgHdr_LineStop' in head:
            LineStop = 2 ** (head['ImgHdr_LineStop'] - 1)
        
        y = []
        tmpx = []
        chan = []
        markers = []
        dt = np.zeros(ny)
        
        im_sync = []
        im_tcspc = []
        im_chan = []
        im_line = []
        im_col = []
        im_frame = []
        Turns1 = []
        Turns2 = []

        cnt = 0
        tend = 0
        line = 0 # python compatible
      
        if head['ImgHdr_BiDirect'] == 0:  # Unidirectional scan
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)
            while num>0:
              
                cnt += num
                if len(y)>0:
                    tmp_sync = tmp_sync + tend
                
                ind = (tmp_special>0) | ((tmp_chan<anzch) & (tmp_tcspc<Ngate*chDiv));
                
                y = np.concatenate((y, tmp_sync[ind]))  # Appending selected elements to y
                tmpx = np.concatenate((tmpx, np.floor(tmp_tcspc[ind] / chDiv) ))  # Appending selected elements to tmpx
                chan = np.concatenate((chan, tmp_chan[ind] ))  # Appending selected elements to chan
                markers = np.concatenate((markers, tmp_special[ind]))  # Appending selected elements to markers

                if LineStart == LineStop:
                    tmpturns = y[markers == LineStart]
                    if len(Turns1) > len(Turns2):
                        Turns1 = np.concatenate([Turns1, tmpturns[1::2]])  # Add elements to Turns1
                        Turns2 = np.concatenate([Turns2, tmpturns[::2]])   # Add elements to Turns2
                    else:
                        Turns1 = np.concatenate([Turns1, tmpturns[::2]])
                        Turns2 = np.concatenate([Turns2, tmpturns[1::2]])
                else:   
                    Turns1 = np.concatenate([Turns1, y[markers == LineStart]])
                    Turns2 = np.concatenate([Turns2, y[markers == LineStop]])

                ind = (markers != 0)
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                chan = np.delete(chan, ind)
                markers = np.delete(markers, ind)

                tend = (y[-1] + loc).astype(np.uint64)

                if len(Turns2) > 1:
                    for j in range(len(Turns2) - 1):
                        t1 = Turns1[0]
                        t2 = Turns2[0]

                        ind = (y < t1)
                        idx = np.where(ind)[0]
                        y = np.delete(y, idx)
                        tmpx = np.delete(tmpx, idx)
                        chan = np.delete(chan, idx)

                        ind0 = (y >= t1) & (y <= t2)
                        ind = np.where(ind0)[0]
                        im_sync.extend(y[ind])
                        im_tcspc.extend(tmpx[ind].astype(np.uint16))
                        im_chan.extend(chan[ind].astype(np.uint8))
                        im_line.extend([line].astype(np.uint16) * np.sum(ind))
                        im_col.extend(np.floor(nx * (y[ind] - t1) / (t2 - t1))) # Python compatible, pixel starts from zero

                        dt[line] = t2 - t1
                        line += 1
                        cwaitbar(line,ny,message = "Lines Processed")

                        Turns1 = Turns1[1:]
                        Turns2 = Turns2[1:]
                tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)
            
            
            t1 = Turns1[-1]
            t2 = Turns2[-1]

            ind          = (y<t1);
            y = np.delete(y, ind)
            tmpx = np.delete(tmpx, ind)
            chan = np.delete(chan, ind)

            ind = (y>=t1) and (y<=t2);

            im_sync.extend(y[ind])
            im_tcspc.extend(tmpx[ind].astype(np.uint16))
            im_chan.extend(chan[ind].astype(np.uint8))
            im_line.extend([line].astype(np.uint16) * np.sum(ind))
            im_col.extend(np.floor(nx * (y[ind] - t1) / (t2 - t1))) # Python compatible, pixel starts from zero
            dt[line]  = t2-t1;
            
            tag, tau, tcspc_pix = Process_Frame(im_sync, im_col, im_line, im_chan, im_tcspc, head, cnum)
            
            line = line +1;    
            head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
            head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']

        else:  # Bidirectional scan
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)
            while num>0:
                cnt += num
                if len(y)>0:
                    tmp_sync = tmp_sync + tend
                
                ind = (tmp_special>0) | ((tmp_chan<anzch) & (tmp_tcspc<Ngate*chDiv));
                
                y = np.concatenate((y, tmp_sync[ind]))  # Appending selected elements to y
                tmpx = np.concatenate((tmpx, np.floor(tmp_tcspc[ind] / chDiv) ))  # Appending selected elements to tmpx
                chan = np.concatenate((chan, tmp_chan[ind] ))  # Appending selected elements to chan
                markers = np.concatenate((markers, tmp_special[ind]))  # Appending selected elements to markers

                if LineStart == LineStop:
                    tmpturns = y[markers == LineStart]
                    if len(Turns1) > len(Turns2):
                        Turns1 = np.concatenate([Turns1, tmpturns[1::2]])  # Add elements to Turns1
                        Turns2 = np.concatenate([Turns2, tmpturns[::2]])   # Add elements to Turns2
                    else:
                        Turns1 = np.concatenate([Turns1, tmpturns[::2]])
                        Turns2 = np.concatenate([Turns2, tmpturns[1::2]])
                else:   
                    Turns1 = np.concatenate([Turns1, y[markers == LineStart]])
                    Turns2 = np.concatenate([Turns2, y[markers == LineStop]])

                ind = (markers != 0)
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                chan = np.delete(chan, ind)
                markers = np.delete(markers, ind)

                tend = y[-1] + loc

                if len(Turns2) > 2:
                    for j in range(0, 2 * (len(Turns2) // 2) - 1, 2):
                        t1 = Turns1[0]
                        t2 = Turns2[0]

                        ind = (y < t1)
                        y = np.delete(y, ind)
                        tmpx = np.delete(tmpx, ind)
                        chan = np.delete(chan, ind)
                        markers = np.delete(markers, ind)

                        ind = (y >= t1) & (y <= t2)
                        idx = np.where(ind)[0]

                        im_sync.extend(y[ind])
                        im_tcspc.extend(tmpx[ind])
                        im_chan.extend(chan[ind])
                        im_line.extend([line] * np.sum(ind))
                        im_col.extend(np.floor(nx * (y[ind] - t1) / (t2 - t1))) # 0 being the first pixel along x direction

                        dt[line] = t2 - t1
                        line += 1

                        t1 = Turns1[1]
                        t2 = Turns2[1]

                        ind = (y < t1)
                        y = np.delete(y, ind)
                        tmpx = np.delete(tmpx, ind)
                        chan = np.delete(chan, ind)
                        markers = np.delete(markers, ind)

                        ind = (y >= t1) & (y <= t2)

                        im_sync.extend(y[ind])
                        im_tcspc.extend(tmpx[ind])
                        im_chan.extend(chan[ind])
                        im_line.extend([line] * np.sum(ind))
                        im_col.extend(nx-1 - np.floor(nx * (y[ind] - t1) / (t2 - t1))) # nx-1 being the last pixel along x direction

                        dt[line] = t2 - t1
                        line += 1
                        cwaitbar(line,ny,message = "Lines Processed")
                        
                        Turns1 = Turns1[2:]
                        Turns2 = Turns2[2:]
                tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)
            
            if len(Turns2) >1:
                t1 = Turns1[-2]
                t2 = Turns2[-2]
                
                ind = (y < t1)
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                chan = np.delete(chan, ind)
                markers = np.delete(markers, ind)

                ind = (y >= t1) & (y <= t2)

                im_sync.extend(y[ind])
                im_tcspc.extend(tmpx[ind])
                im_chan.extend(chan[ind])
                im_line.extend([line] * np.sum(ind))
                im_col.extend(np.floor(nx * (y[ind] - t1) / (t2 - t1))) # 0 being the first pixel along x direction

                dt[line] = t2 - t1
                line += 1

                t1 = Turns1[-1]
                t2 = Turns2[-1]

                ind = (y < t1)
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                chan = np.delete(chan, ind)
                markers = np.delete(markers, ind)

                ind = (y >= t1) & (y <= t2)

                im_sync.extend(y[ind])
                im_tcspc.extend(tmpx[ind])
                im_chan.extend(chan[ind])
                im_line.extend([line] * np.sum(ind))
                im_col.extend(nx-1 - np.floor(nx * (y[ind] - t1) / (t2 - t1))) # nx-1 being the last pixel along x direction

                dt[line] = t2 - t1
                line += 1
                cwaitbar(line,ny,message = "Lines Processed")

            tag, tau, tcspc_pix = Process_Frame(im_sync, im_col, im_line, im_chan, im_tcspc, head, cnum)
             
            head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
            head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']
        
        filename = f"{filename[:-4]}_FLIM_data.pkl"

        # Create a dictionary to store all variables
        data_to_save = {
            'tag': tag,
            'tau': tau,
            'tcspc_pix': tcspc_pix,
            'head': head,
            'im_sync': im_sync,
            'im_tcspc': im_tcspc,
            'im_line': im_line,
            'im_col': im_col,
            'im_chan': im_chan,
            'im_frame': im_frame
            }
        
        # Save the data using pickle
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)    
            
    elif head['ImgHdr_Ident'] == 3:  
        # Common settings
        anzch = 32  # max number of channels (can be 64 now for the new MultiHarp)
        # Resolution = max([1e9 * head['MeasDesc_Resolution'], 0.064])
        Resolution = 1e9*head['MeasDesc_Resolution']
        chDiv = 1e-9 * Resolution / head['MeasDesc_Resolution']
        Ngate = int(np.ceil(1e9 * head['MeasDesc_GlobalResolution'] / Resolution)) + 1
        head['MeasDesc_Resolution'] = Resolution * 1e-9
        
        y = []
        tmpx = []
        chan = []
        marker = []
        
        dt = np.zeros(ny)
        
        # nphot = head['TTResult_NumberOfRecords']
        # im_sync = np.zeros(nphot)
        # im_tcspc = np.zeros(nphot,dtype=np.uint16)
        # im_chan = np.zeros(nphot,dtype=np.uint8)
        # im_line = np.copy(im_tcspc)
        # im_col = np.copy(im_tcspc)
        # im_frame = np.copy(im_tcspc)*np.nan
        
        
        im_sync = []
        im_tcspc = []
        im_chan = []
        im_line = []
        im_col = []
        im_frame = []
        
        cnt = 0
        tend = 0
        line = 0
        cn_phot = 0
        n_frames = -1
        f_times = []
        
        head['ImgHdr_X0'] = 0
        head['ImgHdr_Y0'] = 0
        head['ImgHdr_PixResol'] = 1
        
        LineStart = 2 ** (head['ImgHdr_LineStart'] - 1)
        LineStop = 2 ** (head['ImgHdr_LineStop'] - 1)
        Frame = 2 ** (head['ImgHdr_Frame'] - 1)
        
        in_frame = True;

        if Frame<1:
            Frame = -1
            in_frame = True
            n_frames += 1
          
            
        
        if head['ImgHdr_BiDirect'] == 0: # monodirectional scan
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)    
            
    
            while num > 0:
    
                # t_sync = []
                # t_tcspc = []
                # t_chan = []
                # t_line = []
                # t_col = []
                # t_frame = []
    
                cnt += num
                tmp_sync += tend
                
                
                y = np.concatenate((y, tmp_sync))  # Appending selected elements to y
                tmpx = np.concatenate((tmpx, np.floor(tmp_tcspc / chDiv) ))  # Appending selected elements to tmpx
                chan = np.concatenate((chan, tmp_chan ))  # Appending selected elements to chan
                marker = np.concatenate((marker, tmp_special))  # Appending selected elements to markers
    
    
    
                # y.extend(tmp_sync)
                # tmpx.extend(tmp_tcspc)
                # chan.extend(tmp_chan)
                # marker.extend(tmp_special)
                tend = y[-1] + loc
                
                F = y[(marker.astype(int) & Frame) >0] # Frame changes
    
                # F = [val for val in y if val and Frame > 0]
                while len(F)>0:
    
                    if len(F)>0:
                        ind = (y < F[0])
                        idx = np.where(ind)[0]
                        
                        f_y     = y[idx]
                        f_x     = tmpx[idx]
                        f_ch    = chan[idx]
                        f_m     = marker[idx]
                                          
                                       
                        y = np.delete(y, idx)
                        tmpx = np.delete(tmpx, idx)
                        chan = np.delete(chan, idx)
                        marker = np.delete(marker, idx)
                        
                        L1 = f_y[(f_m.astype(int) & LineStart) > 0]
                        L2 = f_y[(f_m.astype(int) & LineStop) > 0]
                        
                        if (y[0] == F[0]) and (marker[0]==2):
                            L2 = np.concatenate((L2, [y[0]])) 
                            y = np.delete(y, 0)
                            tmpx = np.delete(tmpx, 0)
                            chan = np.delete(chan, 0)
                            marker = np.delete(marker, 0)
                            
                        if (y[0] == F[0]) and (marker[0]==4):
                            y = np.delete(y, 0)
                            tmpx = np.delete(tmpx, 0)
                            chan = np.delete(chan, 0)
                            marker = np.delete(marker, 0)    
                        
                        
                        f_times.append(F[0])
                        F = F[1:]
                        line = 0
    
                        if len(L1) > 1:
                            n_frames += 1
                            for j in range(len(L2)):
                                ind = (f_y > L1[j]) & (f_y < L2[j])
                                idx = np.where(ind)[0]
                                n_phot = len(idx)
                                
                                # im_frame[cn_phot:cn_phot + n_phot ] = np.uint16([n_frames] * np.sum(ind))
                                # im_sync[cn_phot:cn_phot + n_phot ] = f_y[idx]
                                # im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(f_x[idx])
                                # im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(f_ch[idx])
                                # im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                                # im_col[cn_phot:cn_phot + n_phot ] = np.uint16(np.floor(nx * (f_y[idx] - L1[j]) / (L2[j] - L1[j])))
    
                                im_frame.extend( np.uint16([n_frames] * np.sum(ind)))
                                im_sync.extend( f_y[idx])
                                im_tcspc.extend(np.uint16(f_x[idx]))
                                im_chan.extend(np.uint8(f_ch[idx]))
                                im_line.extend( np.uint16(line))
                                im_col.extend(np.uint16(np.floor(nx * (f_y[idx] - L1[j]) / (L2[j] - L1[j]))))                             
                                
                                # t_sync.extend(f_y[idx])
                                # t_tcspc.extend(np.uint16(f_x[idx]))
                                # t_chan.extend(np.uint8(f_ch[idx]))
                                # t_line.extend(np.uint16([line] * np.sum(ind)))
                                # t_col.extend(np.uint16(np.floor(nx * (f_y[idx] - L1[j]) / (L2[j] - L1[j]))))
                                # t_frame.extend(np.uint16([n_frames] * np.sum(ind)))
                                
                                dt[line] += (L2[j] - L1[j])
                                line += 1
    
                        # im_sync.extend(t_sync)
                        # im_tcspc.extend(t_tcspc)
                        # im_chan.extend(t_chan)
                        # im_line.extend(t_line)
                        # im_col.extend(t_col)
                        # im_frame.extend(t_frame)
    
                tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)   
                
            # F = [val for val in y if val & Frame > 0]
            F = y[(marker.astype(int) & Frame) >0] # Frame changes
    
            # t_sync = []
            # t_tcspc = []
            # t_chan = []
            # t_line = []
            # t_col = []
            # t_frame = []
    
            if not in_frame:
                if len(F)==0:
                    y = []
                    tmpx = []
                    chan = []
                    marker = []
                    line = 0
                else:
                    ind = y<=F[0]
                    idx = np.where(ind)[0]
                    y = np.delete(y, idx)
                    tmpx = np.delete(tmpx, idx)
                    chan = np.delete(chan, idx)
                    marker = np.delete(marker, idx)
    
                    line = 0
                    n_frames += 1
                    f_times.append(F[0])
    
            f_y = y
            f_x = tmpx
            f_ch = chan
            f_m = marker
            
            y = []
            tmpx = []
            chan = []
            
            L1 = f_y[(f_m.astype(int) & LineStart) > 0]
            L2 = f_y[(f_m.astype(int) & LineStop) > 0]
            
            
            ll = line + len(L2) - 1
            if ll > ny:
                L1 = L1[:ny - line ]
                L2 = L2[:ny - line ]
    
            if len(L1) > 1:
                for j in range(len(L2)):
                    
                    ind = (f_y > L1[j]) & (f_y < L2[j])
                    idx = np.where(ind)[0]
                    
                    n_phot = len(idx)
                    
                    im_frame.extend( np.uint16([n_frames] * np.sum(ind)))
                    im_sync.extend( f_y[idx])
                    im_tcspc.extend(np.uint16(f_x[idx]))
                    im_chan.extend(np.uint8(f_ch[idx]))
                    im_line.extend( np.uint16(line))
                    im_col.extend(np.uint16(np.floor(nx * (f_y[idx] - L1[j]) / (L2[j] - L1[j]))))
                    
                    # im_frame[cn_phot:cn_phot + n_phot ] = np.uint16([n_frames] * np.sum(ind))
                    # im_sync[cn_phot:cn_phot + n_phot ] = f_y[idx]
                    # im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(f_x[idx])
                    # im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(f_ch[idx])
                    # im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                    # im_col[cn_phot:cn_phot + n_phot ] = np.uint16(np.floor(nx * (f_y[idx] - L1[j]) / (L2[j] - L1[j])))
    
                    # cn_phot += n_phot 
                    
                    
                    # t_sync.extend(f_y[idx])
                    # t_tcspc.extend(np.uint16(f_x[idx]))
                    # t_chan.extend(np.uint8(f_ch[idx]))
                    # t_line.extend(np.uint16([line] * np.sum(ind)))
                    # t_col.extend(np.uint16(np.floor(nx * (f_y[idx] - L1[j]) / (L2[j] - L1[j]))))
                    # t_frame.extend(np.uint16([n_frames] * np.sum(ind)))
                    dt[line] += (L2[j] - L1[j])
                    line += 1
        elif head['ImgHdr_BiDirect'] == 1: # bidirectional scan     
            cnt = 0
            tend = 0
            line = 0 # python compatible
            n_frames = -1
            f_times     = []
            
            y           = []
            tmpx        = []
            chan        = []
            marker      = []
            dt          = np.zeros(ny)
                
            im_sync     = []
            im_tcspc    = []
            im_chan     = []
            im_line     = []
            im_col      = []
            im_frame    = []
            Turns1      = []
            Turns2      = []
            tag         = []
            tau         = []
            
            
            
            tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)    
            dind = np.unique(tmp_chan)
            
    
            while num > 0:
                cnt += num
                tmp_sync += tend
                
                
                y = np.concatenate((y, tmp_sync))  # Appending selected elements to y
                tmpx = np.concatenate((tmpx, np.floor(tmp_tcspc / chDiv) ))  # Appending selected elements to tmpx
                chan = np.concatenate((chan, tmp_chan ))  # Appending selected elements to chan
                marker = np.concatenate((marker, tmp_special))  # Appending selected elements to markers
    
    
                tend = y[-1] + loc
                
                F = y[(marker.astype(int) & Frame) >0] # Frame changes
    
                # F = [val for val in y if val and Frame > 0]
                while len(F)>0:

                    if len(F)>0:
                        ind = (y < F[0])
                        idx = np.where(ind)[0]
                        
                        f_y     = y[idx]
                        f_x     = tmpx[idx]
                        f_ch    = chan[idx]
                        f_m     = marker[idx]
                                          
                                       
                        y = np.delete(y, idx)
                        tmpx = np.delete(tmpx, idx)
                        chan = np.delete(chan, idx)
                        marker = np.delete(marker, idx)
                        
                        L1 = f_y[(f_m.astype(int) & LineStart) > 0]
                        L2 = f_y[(f_m.astype(int) & LineStop) > 0]
                        
                        if (y[0] == F[0]) and (marker[0]==2):
                            L2 = np.concatenate((L2, [y[0]])) 
                            y = np.delete(y, 0)
                            tmpx = np.delete(tmpx, 0)
                            chan = np.delete(chan, 0)
                            marker = np.delete(marker, 0)
                            
                        if (y[0] == F[0]) and (marker[0]==4):
                            y = np.delete(y, 0)
                            tmpx = np.delete(tmpx, 0)
                            chan = np.delete(chan, 0)
                            marker = np.delete(marker, 0)    
                        
                        
                        f_times.append(F[0])
                        F = F[1:]
                        line = 0
                        
                        tmptag = np.zeros((nx,ny,len(dind), cnum))
                        tmptau = np.zeros_like(tmptag)
                        if len(L2) > 2:
                            n_frames += 1
                            for j in range(0, 2 * (len(L2) // 2 - 1), 2):
                                t1 = L1[0]
                                t2 = L2[0]
                                ind = (f_y >= t1) & (f_y <= t2)
                                idx = np.where(ind)[0]
                                n_phot = len(idx)
                                
                                im_frame.extend( np.uint16([n_frames] * np.sum(ind)))
                                im_sync.extend( f_y[idx])
                                im_tcspc.extend(np.uint16(f_x[idx]))
                                im_chan.extend(np.uint8(f_ch[idx]))
                                im_line.extend(np.uint16([line] * np.sum(ind)))
                                im_col.extend(np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1))))
                                
                                tmptag, tmptau = Frame_update(tmptag, tmptau, np.uint16(f_x[idx]), \
                                                              np.uint8(f_ch[idx]), \
                                                              np.uint16(line), \
                                                              np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1))), \
                                                              head, cnum)
                                
                                # im_frame[cn_phot:cn_phot + n_phot ] = np.uint16([n_frames] * np.sum(ind))
                                # im_sync[cn_phot:cn_phot + n_phot ] = f_y[idx]
                                # im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(f_x[idx])
                                # im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(f_ch[idx])
                                # im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                                # im_col[cn_phot:cn_phot + n_phot ] = np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1)))
    
                                # cn_phot += n_phot 
                                
                                dt[line] += t2-t1
                                line += 1
                                
                                t1 = L1[1]
                                t2 = L2[1]
                                
                                ind = (f_y >= t1) & (f_y <= t2)
                                idx = np.where(ind)[0]
                                n_phot = len(idx)
                                
                                im_frame.extend( np.uint16([n_frames] * np.sum(ind)))
                                im_sync.extend( f_y[idx])
                                im_tcspc.extend(np.uint16(f_x[idx]))
                                im_chan.extend(np.uint8(f_ch[idx]))
                                im_line.extend(np.uint16([line] * np.sum(ind)))
                                im_col.extend(np.uint16(nx - np.floor(nx * (f_y[idx] - t1) / (t2 - t1))))
                                
                                tmptag, tmptau = Frame_update(tmptag, tmptau, np.uint16(f_x[idx]), \
                                                              np.uint8(f_ch[idx]), \
                                                              np.uint16(line), \
                                                              np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1))), \
                                                              head, cnum)
                                    
                                # im_frame[cn_phot:cn_phot + n_phot ] = np.uint16([n_frames] * np.sum(ind))
                                # im_sync[cn_phot:cn_phot + n_phot ] = f_y[idx]
                                # im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(f_x[idx])
                                # im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(f_ch[idx])
                                # im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                                # im_col[cn_phot:cn_phot + n_phot ] = np.uint16(nx - np.floor(nx * (f_y[idx] - t1) / (t2 - t1)))
    
                                # cn_phot += n_phot 
                                dt[line] += t2-t1
                                line += 1
                                
                                L1 = L1[2:]
                                L2 = L2[2:]
                        tag.append(tmptag)
                        tau.append(tmptau)
    
                tmp_sync, tmp_tcspc, tmp_chan, tmp_special, num, loc = ptu_reader.get_photon_chunk(cnt+1, photons, head)   
                
            # F = [val for val in y if val & Frame > 0]
            F = y[(marker.astype(int) & Frame) >0] # Frame changes

    
            if not in_frame:
                if len(F)==0:
                    y = []
                    tmpx = []
                    chan = []
                    marker = []
                    line = 0
                else:
                    ind = y<=F[0]
                    idx = np.where(ind)[0]
                    y = np.delete(y, idx)
                    tmpx = np.delete(tmpx, idx)
                    chan = np.delete(chan, idx)
                    marker = np.delete(marker, idx)
    
                    line = 0
                    n_frames += 1
                    f_times.append(F[0])
    
            f_y = y
            f_x = tmpx
            f_ch = chan
            f_m = marker
            
            y = []
            tmpx = []
            chan = []
            
            L1 = f_y[(f_m.astype(int) & LineStart) > 0]
            L2 = f_y[(f_m.astype(int) & LineStop) > 0]
            
            
            ll = line + len(L2) - 1
            if ll > ny:
                L1 = L1[:ny - line ]
                L2 = L2[:ny - line ]
    
            if len(L2) > 2:
                tmptag = np.zeros((nx,ny,len(dind), cnum))
                tmptau = np.zeros_like(tmptag)
                
                for j in range(0, 2 * (len(L2) // 2 - 1), 2):
                    t1 = L1[0]
                    t2 = L2[0]
                    ind = (f_y >= t1) & (f_y <= t2)
                    idx = np.where(ind)[0]
                    n_phot = len(idx)
                    
                    im_frame.extend( np.uint16([n_frames] * np.sum(ind)))
                    im_sync.extend( f_y[idx])
                    im_tcspc.extend(np.uint16(f_x[idx]))
                    im_chan.extend(np.uint8(f_ch[idx]))
                    im_line.extend( np.uint16(line))
                    im_col.extend(np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1))))
                    
                    tmptag, tmptau = Frame_update(tmptag, tmptau, np.uint16(f_x[idx]), \
                                                  np.uint8(f_ch[idx]), \
                                                  np.uint16(line), \
                                                  np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1))), \
                                                  head, cnum)
                    
                    # im_frame[cn_phot:cn_phot + n_phot ] = np.uint16([n_frames] * np.sum(ind))
                    # im_sync[cn_phot:cn_phot + n_phot ] = f_y[idx]
                    # im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(f_x[idx])
                    # im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(f_ch[idx])
                    # im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                    # im_col[cn_phot:cn_phot + n_phot ] = np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1)))

                    # cn_phot += n_phot 
                    
                    dt[line] += t2-t1
                    line += 1
                    
                    t1 = L1[1]
                    t2 = L2[1]
                    
                    ind = (f_y >= t1) & (f_y <= t2)
                    idx = np.where(ind)[0]
                    n_phot = len(idx)
                    
                    im_frame.extend( np.uint16([n_frames] * np.sum(ind)))
                    im_sync.extend( f_y[idx])
                    im_tcspc.extend(np.uint16(f_x[idx]))
                    im_chan.extend(np.uint8(f_ch[idx]))
                    im_line.extend( np.uint16(line))
                    im_col.extend(np.uint16(nx - np.floor(nx * (f_y[idx] - t1) / (t2 - t1))))
                    
                    tmptag, tmptau = Frame_update(tmptag, tmptau, np.uint16(f_x[idx]), \
                                                  np.uint8(f_ch[idx]), \
                                                  np.uint16(line), \
                                                  np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1))), \
                                                  head, cnum)
                    # im_frame[cn_phot:cn_phot + n_phot ] = np.uint16([n_frames] * np.sum(ind))
                    # im_sync[cn_phot:cn_phot + n_phot ] = f_y[idx]
                    # im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(f_x[idx])
                    # im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(f_ch[idx])
                    # im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                    # im_col[cn_phot:cn_phot + n_phot ] = np.uint16(np.floor(nx * (f_y[idx] - t1) / (t2 - t1)))

                    # cn_phot += n_phot 
                    dt[line] += t2-t1
                    line += 1
                    
                    L1 = L1[2:]
                    L2 = L2[2:]
                tag.append(tmptag)
                tau.appent(tmptau)

        head['ImgHdr_FrameTime'] = 1e9 * np.mean(np.diff(f_times)) / head['TTResult_SyncRate']
        head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
        head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime'] / n_frames
        
        # dind = np.unique(im_chan)
        # tag = np.zeros((nx,ny,len(dind),np.max(im_frame)+1, cnum))
        # tau = np.copy(tag)
        im_frame = np.asarray(im_frame)
        im_col = np.asarray(im_col)
        im_sync = np.asarray(im_sync)
        im_line = np.asarray(im_line)
        im_chan = np.asarray(im_chan)
        im_tcspc = np.asarray(im_tcspc)
        
        tag = np.transpoer(np.asarray(tag),(1,2,3,0,4)) # xyctp
        tau = np.transpoer(np.asarray(tau),(1,2,3,0,4)) # xyctp
        # print('Processing Frames')
        
        # for frame in tqdm(range(np.max(im_frame)),desc= 'Processing Frames:'):
        #     print('Fast')
        #     tmptag, tmptau,_ = Process_FrameFast(im_sync[im_frame == frame],im_col[im_frame == frame],\
        #                                      im_line[im_frame == frame],im_chan[im_frame == frame],\
        #                                          im_tcspc[im_frame == frame],head, cnum, resolution = 0.2)
        #     if tmptag is not None and tmptag.size > 0:
        #         tag[:,:,:,frame, :] = tmptag
        #         tau[:,:,:,frame, :] = tmptau
        
        SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
        maxch_n = len(dind)
        
        tcspc_pix = np.zeros((nx, ny, Ngate, maxch_n*cnum), dtype=np.uint32)
        timeF = [None] * maxch_n*cnum  # Initialize a list to store time data for each channel
        tags = np.zeros((nx, ny, maxch_n,cnum), dtype=np.uint32)
        taus = np.zeros((nx, ny, maxch_n,cnum))
        binT = np.transpose(np.tile(np.arange(1, Ngate + 1).reshape(-1, 1, 1) * Resolution, (1, nx, ny)), (1, 2, 0))  # 3D time axis
        
        tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
        
        for ch in range(maxch_n):
            for p in range(cnum):
                ind = (im_chan == dind[ch]) & (im_tcspc<tmpCh/cnum*(p+1)) & (im_tcspc>=p*tmpCh/cnum)
                idx = np.where(ind)[0]
                # print(len(idx))
                tcspc_pix[:, :, :, ch*cnum + p] = mHist3(im_line[idx].astype(np.int64), 
                                            im_col[idx].astype(np.int64), 
                                            (im_tcspc[idx] / chDiv).astype(np.int64) - int(p*Ngate/cnum), 
                                            np.arange(nx), 
                                            np.arange(ny), 
                                            np.arange(Ngate))[0]  # tcspc histograms for all the pixels at once!
            
               
                tag[:, :, ch, p] = np.expand_dims(np.sum(tcspc_pix[:, :, :, ch*cnum + p], axis=2), axis = -1)
                tau[:, :, ch, p] = np.expand_dims(np.real(np.sqrt((np.sum(binT ** 2 * tcspc_pix[:, :, :, ch*cnum + p], axis=2) / (np.sum(tag[:, :, ch, p], axis = -1) + 10**-10)) -
                                                (np.sum(binT * tcspc_pix[:, :, :, ch*cnum + p], axis=2) / (np.sum(tag[:, :, ch, p], axis =-1) + 10**-10)) ** 2)), axis = -1)
                timeF[ch*cnum + p] = np.round(im_sync[idx] / SyncRate / Resolution / 1e-9) + im_tcspc[idx].astype(np.int64)  # in tcspc bins 
                
        # for ch in range(maxch_n):
        #     ind = (im_chan == dind[ch] + 1)
        #     tcspc_pix[:, :, :, ch] = mHist3(im_line[ind].astype(np.uint64), 
        #                                     im_col[ind].astype(np.uint64), 
        #                                     im_tcspc[ind].astype(np.uint64), 
        #                                     np.arange(nx), 
        #                                     np.arange(ny), 
        #                                     np.arange(Ngate))
            
        #     timeF[ch] = np.round(im_sync[ind] / SyncRate / Resolution / 1e-9) + im_tcspc[ind].astype(float)
        #     tags[:, :, ch] = np.sum(tcspc_pix[:, :, :, ch], axis=2)
        #     taus[:, :, ch] = np.real(np.sqrt((np.sum(binT ** 2 * tcspc_pix[:, :, :, ch], axis=2) / tags[:, :, ch]) -
        #                                      (np.sum(binT * tcspc_pix[:, :, :, ch], axis=2) / tags[:, :, ch]) ** 2))
            
            
        filename = f"{filename[:-4]}_FLIM_data.pkl"

        # Create a dictionary to store all variables
        data_to_save = {
            'tag': tag,
            'tau': tau,
            'tags': tags,
            'taus': taus,
            'time': timeF,
            'tcspc_pix': tcspc_pix,
            'head': head,
            'im_sync': im_sync,
            'im_tcspc': im_tcspc,
            'im_line': im_line,
            'im_col': im_col,
            'im_chan': im_chan,
            'im_frame': im_frame
            }
        
        # Save the data using pickle
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)  
            
            
    elif head['ImgHdr_Ident'] == 9:          # ------------------------------------Multiframe scan------------------------
        if 'ImgHdr_MaxFrames' in head:
            nz = head['ImgHdr_MaxFrames']
        else:
            tim_p_frame = 1/head['ImgHdr_LineFrequency']/ny
            tot_time = head['TTResult_StopAfter']*10^-3
            nz = np.ceil(tot_time/tim_p_frame)
        
        if 'PIENumPIEWindows' in head:
            cnum = head['PIENumPIEWindows'] # number of PIE cycles - this effects the tcspc channels
        anzch = 32  # max number of channels (can be 64 now for the new MultiHarp)
        # Resolution = max([1e9 * head['MeasDesc_Resolution'], 0.064])
        Resolution = 1e9*head['MeasDesc_Resolution']
        chDiv = 1e-9 * Resolution / head['MeasDesc_Resolution']
        Ngate = int(np.ceil(1e9 * head['MeasDesc_GlobalResolution'] / Resolution))
        head['MeasDesc_Resolution'] = Resolution * 1e-9
        
        sync, tcspc, tmpchan, tmpmarkers, num, loc = ptu_reader.get_photon_chunk(1, 10^4, head)
        dind = np.unique([val for i, val in enumerate(tmpchan) if not tmpmarkers[i]]) # the number of detectors
        tag = np.zeros((nx, ny, len(dind), nz, cnum))
        tau = np.copy(tag)
        
        y = []
        tmpx = []
        chan = []
        markers = []
        
        dt = np.zeros(ny)
        nphot = head['TTResult_NumberOfRecords']
        im_sync = np.zeros(nphot)
        im_tcspc = np.zeros(nphot,dtype=np.uint16)
        im_chan = np.zeros(nphot,dtype=np.uint8)
        im_line = np.copy(im_tcspc)
        im_col = np.copy(im_tcspc)
        im_frame = np.copy(im_tcspc)*np.nan
        
        cnt = 0
        tend = 0
        line = 0
        frame = 0
        cn_phot = 0
        Turns1 = []
        Turns2 = []
        
        
        LineStart = 4
        LineStop = 2
        Frame = 3
        if 'ImgHdr_LineStart' in head:
            LineStart = 2 ** (head['ImgHdr_LineStart'] - 1)
        if 'ImgHdr_LineStop' in head:
            LineStop = 2 ** (head['ImgHdr_LineStop'] - 1)
        if 'ImgHdr_Frame' in head:    
            Frame = 2 ** (head['ImgHdr_Frame'] - 1)
            
        if head['ImgHdr_BiDirect'] == 0: # monodirectional scan
            tmpy, tmptcspc, tmpchan, tmpmarkers, num, loc =  ptu_reader.get_photon_chunk(cnt+1, photons, head)
            
            while num>0:
                cnt += num
                if len(y)>0:
                    tmpy += tend
                    
                ind = (tmpmarkers>0) | ((tmpchan<anzch) & (tmptcspc<Ngate*chDiv));
                idx = np.where(ind)[0]
                y = np.concatenate((y, tmpy[idx])).astype(np.uint64)  # Appending selected elements to y
                tmpx = np.concatenate((tmpx, np.floor(tmptcspc[idx] / chDiv) )).astype(np.uint64)   # Appending selected elements to tmpx
                chan = np.concatenate((chan, tmpchan[idx] )).astype(np.uint64)   # Appending selected elements to chan
                markers = np.concatenate((markers, tmpmarkers[idx])).astype(np.uint64)   # Appending selected elements to markers

                if LineStart == LineStop:
                    tmpturns = y[markers == LineStart]
                    if len(Turns1) > len(Turns2):
                        Turns1 = np.concatenate([Turns1, tmpturns[1::2]])  # Add elements to Turns1
                        Turns2 = np.concatenate([Turns2, tmpturns[::2]])   # Add elements to Turns2
                    else:
                        Turns1 = np.concatenate([Turns1, tmpturns[::2]])
                        Turns2 = np.concatenate([Turns2, tmpturns[1::2]])
                else:   
                    Turns1 = np.concatenate([Turns1, y[markers == LineStart]])
                    Turns2 = np.concatenate([Turns2, y[markers == LineStop]])
                
                Framechange = y[markers == np.uint8(Frame)]    
                ind = np.where(markers != 0)[0]
                y = np.delete(y, ind)
                tmpx = np.delete(tmpx, ind)
                chan = np.delete(chan, ind)
                markers = np.delete(markers, ind)

                tend = (y[-1] + loc).astype(np.uint64) # making sure tend is not a float!
            
                if len(Framechange)>=1:
                    for k in range(len(Framechange)):
                        line = 0
                        ind = y<Framechange[k]
                        idx = np.where(ind)[0]
                        yf = y[idx]
                        tmpxf = tmpx[idx]
                        chanf = chan[idx]
                        
                        # y = np.delete(y, ind)
                        y = y[~ind]
                        tmpx = tmpx[~ind]
                        chan = chan[~ind]
                        markers = markers[~ind]
                        # y = np.delete(y,idx)
                        # tmpx = np.delete(tmpx,idx)
                        # chan = np.delete(chan,idx)
                        # markers = np.delete(markers,idx)
                      
                        Turns2f = Turns2[Turns2<Framechange[k]]
                        Turns1f = Turns1[Turns1<Framechange[k]]
                        Turns2 = np.delete(Turns2,Turns2<Framechange[k])
                        Turns1 = np.delete(Turns1,Turns1<Framechange[k])
                        
                        if len(Turns2f) > 1:
                            for j in range(len(Turns2f)):

                                t1 = Turns1f[0]
                                t2 = Turns2f[0]

                                # ind = np.where(~(yf <= t1))[0]
                                # yf = yf[ind]
                                # tmpxf = tmpxf[ind]
                                # chanf = chanf[ind]
                                # markersf = markersf[ind]  # Uncomment if markersf is used

                                ind = (yf > t1) & (yf < t2)
                                idx = np.where(ind)[0]
                                n_phot = len(idx)
                                
                                im_frame[cn_phot:cn_phot + n_phot ] = np.uint16(frame )
                                im_sync[cn_phot:cn_phot + n_phot ] = yf[idx]
                                im_tcspc[cn_phot:cn_phot + n_phot ] = np.uint16(tmpxf[idx])
                                im_chan[cn_phot:cn_phot + n_phot ] = np.uint8(chanf[idx])
                                im_line[cn_phot:cn_phot + n_phot ] = np.uint16(line)
                                im_col[cn_phot:cn_phot + n_phot ] = np.uint16(np.floor(nx * (yf[idx] - t1) / (t2 - t1)))

                                cn_phot += n_phot 
                                
                                dt[line] = t2 - t1
                                line += 1

                                Turns1f = Turns1f[1:]
                                Turns2f = Turns2f[1:]
                                # cwaitbar(line,ny,message = "Lines Processed")
                        idx = np.where(im_frame == frame)[0]
                        tmptag, tmptau,_ = Process_Frame(im_sync[idx],im_col[idx],\
                                                         im_line[idx],im_chan[idx],\
                                                             im_tcspc[idx], head, cnum)
                        if tmptag.size > 0:
                            tag[:,:,:,frame, :] = tmptag
                            tau[:,:,:,frame, :] = tmptau
                        
                        frame += 1
                        cwaitbar(frame,nz,message = "Fames Processed")
                tmpy, tmptcspc, tmpchan, tmpmarkers, num, loc =  ptu_reader.get_photon_chunk(cnt+1, photons, head)
            
            head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
            head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']    
            im_frame = im_frame[:cn_phot]
            im_sync = im_sync[:cn_phot]
            im_tcspc = im_tcspc[:cn_phot]
            im_chan = im_chan[:cn_phot]
            im_line = im_line[:cn_phot]
            im_col = im_col[:cn_phot]
               
        elif head['ImgHdr_BiDirect'] == 1: # bidirectional scan           
             tmpy, tmptcspc, tmpchan, tmpmarkers, num, loc =  ptu_reader.get_photon_chunk(cnt+1, photons, head) 
             while num>0:
                 cnt += num
                 if len(y)>0:
                     tmpy += tend
                     
                 ind = (tmpchan < anzch) & (tmptcspc <= Ngate * chDiv)

                 y = np.concatenate([y, tmpy[ind]])
                 tmpx = np.concatenate([tmpx, np.floor(tmptcspc[ind] / chDiv).astype(int) ])
                 chan = np.concatenate([chan, tmpchan[ind]])
                 markers = np.concatenate([markers, tmpmarkers[ind]])

                 if LineStart == LineStop:
                     tmpturns = y[markers == LineStart]
                     if len(Turns1) > len(Turns2):  # first turn is a LineStop
                         Turns1 = np.concatenate([Turns1, tmpturns[1::2]])  # select even indices
                         Turns2 = np.concatenate([Turns2, tmpturns[0::2]])  # select odd indices
                     else:
                         Turns1 = np.concatenate([Turns1, tmpturns[0::2]])  # select odd indices
                         Turns2 = np.concatenate([Turns2, tmpturns[1::2]])  # select even indices
                 else:
                     Turns1 = np.concatenate([Turns1, y[markers == LineStart]])
                     Turns2 = np.concatenate([Turns2, y[markers == LineStop]])

                 Framechange = y[markers == Frame]

                 ind = (markers != 0)
                 y = y[~ind]
                 tmpx = tmpx[~ind]
                 chan = chan[~ind]
                 markers = markers[~ind]
                 
                 tend = y[-1] + loc
                 
                 if len(Framechange) >= 1:
                     for k in range(len(Framechange)):
                         line = 0
                         ind = y < Framechange[k]
                         
                         yf = y[ind]
                         tmpxf = tmpx[ind]
                         chanf = chan[ind]
                         markersf = markers[ind]
                         
                         y = y[~ind]
                         tmpx = tmpx[~ind]
                         chan = chan[~ind]
                         markers = markers[~ind]
                         
                         Turns2f = Turns2[Turns2 <= Framechange[k]]
                         Turns1f = Turns1[Turns1 <= Framechange[k]]
                         Turns2 = Turns2[Turns2 > Framechange[k]]
                         Turns1 = Turns1[Turns1 > Framechange[k]]
                         
                         if len(Turns2f) > 2:
                             for j in range(0, 2 * (len(Turns2f) // 2 - 1), 2):
                                 t1 = Turns1f[0]
                                 t2 = Turns2f[0]
                                 
                                 # ind = (yf < t1)
                                 # yf = yf[~ind]
                                 # tmpxf = tmpxf[~ind]
                                 # chanf = chanf[~ind]
                                 # markersf = markersf[~ind]
                                 
                                 ind = (yf >= t1) & (yf <= t2)
                                 
                                 im_frame[cn_phot:cn_phot + np.sum(ind)] = np.uint16(frame)
                                 im_sync[cn_phot:cn_phot + np.sum(ind)] = yf[ind]
                                 im_tcspc[cn_phot:cn_phot + np.sum(ind)] = np.uint16(tmpxf[ind])
                                 im_chan[cn_phot:cn_phot + np.sum(ind)] = np.uint8(chanf[ind])
                                 im_line[cn_phot:cn_phot + np.sum(ind)] = np.uint16(line)
                                 im_col[cn_phot:cn_phot + np.sum(ind)] = np.uint16(np.floor(nx * (yf[ind] - t1) / (t2 - t1)))
                                 
                                 cn_phot += np.sum(ind)
                                 dt[line] = t2 - t1
                                 line += 1
                                 
                                 t1 = Turns1f[1] 
                                 t2 = Turns2f[1]
                                 
                                 ind = (yf < t1)
                                 yf = yf[~ind]
                                 tmpxf = tmpxf[~ind]
                                 chanf = chanf[~ind]
                                 markersf = markersf[~ind]
                                 
                                 ind = (yf >= t1) & (yf <= t2)
                                 
                                 im_frame[cn_phot:cn_phot + np.sum(ind)] = np.uint16(frame)
                                 im_sync[cn_phot:cn_phot + np.sum(ind)] = yf[ind]
                                 im_tcspc[cn_phot:cn_phot + np.sum(ind)] = np.uint16(tmpxf[ind])
                                 im_chan[cn_phot:cn_phot + np.sum(ind)] = np.uint8(chanf[ind])
                                 im_line[cn_phot:cn_phot + np.sum(ind)] = np.uint16(line)
                                 im_col[cn_phot:cn_phot + np.sum(ind)] = np.uint16(nx - np.floor(nx * (yf[ind] - t1) / (t2 - t1)))
                                 
                                 cn_phot += np.sum(ind)
                                 dt[line] = t2 - t1
                                 line += 1
                                 
                                 Turns1f = Turns1f[2:]
                                 Turns2f = Turns2f[2:]
                                 
                                 
                         # Process the frame data
                         tmptag, tmptau, _ = Process_Frame(im_sync[im_frame == frame], \
                                                        im_col[im_frame == frame], im_line[im_frame == frame], \
                                                            im_chan[im_frame == frame], im_tcspc[im_frame == frame], \
                                                                head, cnum)
                         if tmptag.size > 0:
                             tag[:, :, :, frame, :] = tmptag
                             tau[:, :, :, frame, :] = tmptau
                         
                         frame += 1
                         cwaitbar(frame,nz,message = "Fames Processed")
                    # Read the next chunk of data
                 tmpy, tmptcspc, tmpchan, tmpmarkers, num, loc =  ptu_reader.get_photon_chunk(cnt+1, photons,head)
             if len(Turns2) > 0:
                 t1 = Turns1[-2]
                 t2 = Turns2[-2]
                 
                 ind = y < t1
                 y = y[~ind]
                 tmpx = tmpx[~ind]
                 chan = chan[~ind]
                 
                 ind = (y >= t1) & (y <= t2)
                 
                 im_frame[cn_phot:cn_phot + np.sum(ind)] = np.uint16(frame)
                 im_sync[cn_phot:cn_phot + np.sum(ind)] = y[ind]
                 im_tcspc[cn_phot:cn_phot + np.sum(ind)] = np.uint16(tmpx[ind])
                 im_chan[cn_phot:cn_phot + np.sum(ind)] = np.uint8(chan[ind])
                 im_line[cn_phot:cn_phot + np.sum(ind)] = np.uint16(line)
                 im_col[cn_phot:cn_phot + np.sum(ind)] = np.uint16(np.floor(nx * (y[ind] - t1) / (t2 - t1)))
                 dt[line] = t2 - t1
                 
                 line += 1
                 cn_phot += np.sum(ind)
                 
                 t1 = Turns1[-1]
                 t2 = Turns2[-1]
                 
                 ind = y < t1
                 y = y[~ind]
                 tmpx = tmpx[~ind]
                 chan = chan[~ind]
                 
                 ind = (y >= t1) & (y <= t2)
                 
                 im_frame[cn_phot:cn_phot + np.sum(ind)] = np.uint16(frame)
                 im_sync[cn_phot:cn_phot + np.sum(ind)] = y[ind]
                 im_tcspc[cn_phot:cn_phot + np.sum(ind)] = np.uint16(tmpx[ind])
                 im_chan[cn_phot:cn_phot + np.sum(ind)] = np.uint8(chan[ind])
                 im_line[cn_phot:cn_phot + np.sum(ind)] = np.uint16(line)
                 im_col[cn_phot:cn_phot + np.sum(ind)] = np.uint16(nx - np.floor(nx * (y[ind] - t1) / (t2 - t1)))
                 
                 cn_phot += np.sum(ind)
                 dt[line] = t2 - t1
                 
                 line += 1

                
            
             tmptag, tmptau,_ = Process_Frame(im_sync[im_frame == frame], im_col[im_frame == frame],\
                                            im_line[im_frame == frame], im_chan[im_frame == frame],\
                                                im_tcspc[im_frame == frame], head, cnum)
             if frame <= nz and tmptag.size > 0:
                 tag[:, :, :, frame, :] = tmptag
                 tau[:, :, :, frame, :] = tmptau

             head['ImgHdr_PixelTime'] = 1e9 * np.mean(dt) / nx / head['TTResult_SyncRate']
             head['ImgHdr_DwellTime'] = head['ImgHdr_PixelTime']
             # Trim arrays to the current number of photons
             im_frame = im_frame[:cn_phot]
             im_sync = im_sync[:cn_phot]
             im_tcspc = im_tcspc[:cn_phot]
             im_chan = im_chan[:cn_phot]
             im_line = im_line[:cn_phot]
             im_col = im_col[:cn_phot]
            
              
        Resolution = max(head['MeasDesc_Resolution'] * 1e9, 0.256)  # resolution of 0.256 ns to calculate average lifetimes
        chDiv = 1e-9 * Resolution / head['MeasDesc_Resolution']
        SyncRate = 1.0 / head['MeasDesc_GlobalResolution']
        nx = head['ImgHdr_PixX']
        ny = head['ImgHdr_PixY']
        dind = np.unique(im_chan).astype(np.int64)
        Ngate = round(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution'] * (head['MeasDesc_Resolution'] / Resolution / cnum) * 1e9)
        tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
        maxch_n = len(dind)


        
        tcspc_pix = np.zeros((nx, ny, Ngate, maxch_n*cnum), dtype=np.uint32)
        timeF = [None] * maxch_n*cnum  # Initialize a list to store time data for each channel
        tags = np.zeros((nx, ny, maxch_n,cnum), dtype=np.uint32)
        taus = np.zeros((nx, ny, maxch_n,cnum))
        binT = np.transpose(np.tile(np.arange(Ngate).reshape(-1, 1, 1) * Resolution, (1, nx, ny)), (1, 2, 0))  # 3D time axis
        
        # for ch in range(maxch_n):
        #     ind = (im_chan == dind[ch])
        #     tcspc_pix[:, :, :, ch] = mHist3(im_line[ind].astype(float), 
        #                                     im_col[ind].astype(float), 
        #                                     im_tcspc[ind].astype(float), 
        #                                     np.arange(nx), 
        #                                     np.arange(ny), 
        #                                     np.arange(Ngate))[0]
            
        #     timeF[ch] = np.round(im_sync[ind] / SyncRate / Resolution / 1e-9) + im_tcspc[ind].astype(float)
        #     tags[:, :, ch] = np.sum(tcspc_pix[:, :, :, ch], axis=2)
        #     taus[:, :, ch] = np.real(np.sqrt((np.sum(binT ** 2 * tcspc_pix[:, :, :, ch], axis=2) / tags[:, :, ch]) -
        #                                      (np.sum(binT * tcspc_pix[:, :, :, ch], axis=2) / tags[:, :, ch]) ** 2))
        tmpCh = np.ceil(head['MeasDesc_GlobalResolution'] / head['MeasDesc_Resolution']) # total number of channels in the original tcspc histogram
        
        for ch in range(maxch_n):
            for p in range(cnum):
                ind = (im_chan == dind[ch]) & (im_tcspc<tmpCh/cnum*(p+1)) & (im_tcspc>=p*tmpCh/cnum)
                idx = np.where(ind)[0]
                # print(len(idx))
                tcspc_pix[:, :, :, ch*cnum + p] = mHist3(im_line[idx].astype(np.int64), 
                                            im_col[idx].astype(np.int64), 
                                            (im_tcspc[idx] / chDiv).astype(np.int64) - int(p*Ngate/cnum), 
                                            np.arange(nx), 
                                            np.arange(ny), 
                                            np.arange(Ngate))[0]  # tcspc histograms for all the pixels at once!
            
               
                tag[:, :, ch, p] = np.expand_dims(np.sum(tcspc_pix[:, :, :, ch*cnum + p], axis=2), axis = -1)
                tau[:, :, ch, p] = np.expand_dims(np.real(np.sqrt((np.sum(binT ** 2 * tcspc_pix[:, :, :, ch*cnum + p], axis=2) / (np.sum(tag[:, :, ch, p], axis = -1) + 10**-10)) -
                                               (np.sum(binT * tcspc_pix[:, :, :, ch*cnum + p], axis=2) / (np.sum(tag[:, :, ch, p], axis =-1) + 10**-10)) ** 2)), axis = -1)
                timeF[ch*cnum + p] = np.round(im_sync[idx] / SyncRate / Resolution / 1e-9) + im_tcspc[idx].astype(np.int64)  # in tcspc bins 
                
            
            
        filename = f"{filename[:-4]}_FLIM_data.pkl"

        # Create a dictionary to store all variables
        data_to_save = {
            'tag': tag,
            'tau': tau,
            'tags': tags,
            'taus': taus,
            'time': timeF,
            'tcspc_pix': tcspc_pix,
            'head': head,
            'im_sync': im_sync,
            'im_tcspc': im_tcspc,
            'im_line': im_line,
            'im_col': im_col,
            'im_chan': im_chan,
            'im_frame': im_frame
            }
        
        # Save the data using pickle
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)    
            
            
            
      
    else:
        print("Unsupported ImgHdr_Ident value:", head['ImgHdr_Ident'])
        return None, None, None, None, None, None, None

    # Plot if required
    if plt_flag:
        x = head['ImgHdr_X0'] + np.arange(nx) * head['ImgHdr_PixResol']
        y = head['ImgHdr_Y0'] + np.arange(ny) * head['ImgHdr_PixResol']

        tagT = np.sum(tags, axis=tuple(range(2, tags.ndim)))
        tauT = np.sum(taus, axis=tuple(range(2, taus.ndim)))

        plt.figure()
        plt.imshow(tagT, extent=(x.min(), x.max(), y.min(), y.max()))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.xlabel('x / µm')
        plt.ylabel('y / µm')
        plt.title('Intensity')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(tauT, extent=(x.min(), x.max(), y.min(), y.max()))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.xlabel('x / µm')
        plt.ylabel('y / µm')
        plt.title('FLIM')
        plt.colorbar()
        plt.show()

    return head, np.array(im_sync), np.array(im_tcspc), np.array(im_chan), np.array(im_line), np.array(im_col), np.array(im_frame)
    

#%%    
if __name__ == '__main__':
    PTU_ScanRead(r'D:\Collabs\fromMarcel\ATS6.ptu', cnum=1, plt_flag=False)    