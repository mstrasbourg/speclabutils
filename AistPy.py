import numpy as np
from UsefulUtils import Bunch
import csv
from pathlib import Path
# Function that import data from text files

def loadtxt_2D(fname, signal='Height(sen)'):
    """Loads a 2D map that was exported to text from the AIST software.

    Parameters
    ----------
    fname : str
        The file to load.

    signal : str, optional
        The signal (i.e., channel) that is contained in the map. Common values might be 
        'Height(sen)', 'mag', 'phase', etc. Default is 'Height(sen)'.

    Returns
    -------
    Bunch    
    """    
    with open(fname) as f:
        iterator = csv.reader(f, delimiter = '\t')
        B = Bunch()
        num_rows, num_cols = 0, 0
        for row in iterator:
            if row[0][0].isalpha():
                key, val = tuple(row[0].split(' = '))
                B.__dict__[key] = val
            else:
                if int(row[1]) > num_rows: num_rows = int(row[1])
                if int(row[0]) > num_cols: num_cols = int(row[0])
        map = np.zeros((num_rows + 1, num_cols + 1))
        f.seek(0)
        for row in iterator:
            if row[0][0].isalpha():
                continue
            else:
#                        y            x              z
                map[int(row[1]), int(row[0])] = float(row[2])
    
        B.map = map
        B.filename = fname
        x0, y0 = float(B.XOrigin), float(B.YOrigin)
        xs, ys = float(B.XSize), float(B.YSize)
        B.extent_olower = [x0, x0 + xs, y0, y0 + ys]
        B.xlocs = np.linspace(x0, x0 + xs, int(B.XPoints))
        B.ylocs = np.linspace(y0, y0 + ys, int(B.YPoints))
        B.scan_num = int(B.FileName.split(' ')[1])
        
    return B

def loadtxt_spec_stack(fname):
    """Loads a spec_stack measurement that was exported to text from the AIST software.

    Parameters
    ----------
    fname : str
        The file to load.

    Returns
    -------
    Bunch    
    """
    B = Bunch(fname=fname)
    with open(fname, 'r') as file:
        for row in file:
            try:
                float(row[0])
            except:
                key, value = row.split(' = ')
                B.__dict__[key] = str(value)[:-1]
        B.spec_data=np.zeros((int(B.YPoints), int(B.XPoints)))
        file.seek(0)
        for row in file:
            try:
                _ = row.split('\t')
                i, j, data = int(_[0]), int(_[1]), float(_[2])
                B.spec_data[j, i] = data
            except:
                continue
                
        x0 = float(B.XOrigin)
        xf = x0 + float(B.XSize)
        
        B.wns = np.linspace(x0, xf, int(B.XPoints))
        B.wls = (1/632.78 - 1e-7*B.wns)**-1
        B.integrated_ints = np.sum(B.spec_data, axis=1)
        B.scan_num = int(Path(B.fname).name.split(' ')[1])
        B.use = True
        
    return B 

def load_curve(fname):
    """Loads a frequency response curve that has been exported to .txt using the AIST software.

    Parameters
    ----------
    fname : str
        The file to load.

    Returns
    -------
    Bunch
    """
    B = Bunch(fname=fname)

    with open(fname) as f:
        B.x = []
        B.y = []
        iterator = csv.reader(f, delimiter = '\t')
        for num, ln in enumerate(iterator):
            # _ = ln.split('\t')
            # print(_)
            if num == 0:
                B.xheader = str(ln[0])
                B.yheader = str(ln[-1][:-2])

            else:
                B.x.append(float(ln[0]))
                B.y.append(float(ln[-1]))
    return B