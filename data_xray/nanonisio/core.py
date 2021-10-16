#sourced from nanonispy by underchemist
#modified by Petro Maksymovych to: (1) make context aware header parser;
#                                   (2) package data to xarray
#final modification October 15, 2017

# from ..modules import *
import os
import xarray as xr
import numpy as np
import pyperclip
import pandas as pd


_end_tags = dict(grid=':HEADER_END:', scan='SCANIT_END', spec='[DATA]')

class NanonisFile:

    """
    Base class for Nanonis data files (grid, scan, point spectroscopy).
    Handles methods and parsing tasks common to all Nanonis files.
    Parameters
    ----------
    fname : str
        Name of Nanonis file.
    Attributes
    ----------
    datadir : str
        Directory path for Nanonis file.
    basename : str
        Just the filename, no path.
    fname : str
        Full path of Nanonis file.
    filetype : str
        filetype corresponding to filename extension.
    byte_offset : int
        Size of header in bytes.
    header_raw : str
        Unproccessed header information.
    """

    def __init__(self, fname):
        self.datadir, self.basename = os.path.split(fname)
        self.fname = fname
        self.filetype = self._determine_filetype()
        self.byte_offset = self.start_byte()
        self.header_raw = self.read_raw_header(self.byte_offset)

    def _determine_filetype(self):
        """
        Check last three characters for appropriate file extension,
        raise error if not.
        Returns
        -------
        str
            Filetype name associated with extension.
        Raises
        ------
        UnhandledFileError
            If last three characters of filename are not one of '3ds',
            'sxm', or 'dat'.
        """

        if self.fname[-3:] == '3ds':
            return 'grid'
        elif self.fname[-3:] == 'sxm':
            return 'scan'
        elif self.fname[-3:] == 'dat':
            return 'spec'
        else:
            raise UnhandledFileError('{} is not a supported filetype or does not exist'.format(self.basename))

    def read_raw_header(self, byte_offset):
        """
        Return header as a raw string.
        Everything before the end tag is considered to be part of the header.
        the parsing will be done later by subclass methods.
        Parameters
        ----------
        byte_offset : int
            Size of header in bytes. Read up to this point in file.
        Returns
        -------
        str
            Contents of filename up to byte_offset as a decoded binary
            string.
        """

        with open(self.fname, 'rb') as f:
            return f.read(byte_offset).decode()

    def start_byte(self):
        """
        Find first byte after end tag signalling end of header info.
        Caveat, I believe this is the first byte after the end of the
        line that the end tag is found on, not strictly the first byte
        directly after the end tag is found. For example in Scan
        __init__, byte_offset is incremented by 4 to account for a
        'start' byte that is not actual data.
        Returns
        -------
        int
            Size of header in bytes.
        """

        with open(self.fname, 'rb') as f:
            tag = _end_tags[self.filetype]

            # Set to a default value to know if end_tag wasn't found
            byte_offset = -1

            for line in f:
                # Convert from bytes to str
                entry = line.strip().decode()
                if tag in entry:
                    byte_offset = f.tell()
                    break

            if byte_offset == -1:
                raise FileHeaderNotFoundError(
                        'Could not find the {} end tag in {}'.format(tag, self.basename)
                        )

        return byte_offset

class Grid(NanonisFile):

    """
    Nanonis grid file class.
    Contains data loading method specific to Nanonis grid file. Nanonis
    3ds files contain a header terminated by '\r\n:HEADER_END:\r\n'
    line, after which big endian encoded binary data starts. A grid is
    always recorded in an 'up' direction, and data is recorded
    sequentially starting from the first pixel. The number of bytes
    corresponding to a single pixel will depend on the experiment
    parameters. In general the size of one pixel will be a sum of
        - # fixed parameters
        - # experimental parameters
        - # sweep signal points (typically bias).
    Hence if there are 2 fixed parameters, 8 experimental parameters,
    and a 512 point bias sweep, a pixel will account 4 x (522) = 2088
    bytes of data. The class intuits this from header info and extracts
    the data for you and cuts it up into each channel, though normally
    this should be just the current.
    Currently cannot accept grids that are incomplete.
    Parameters
    ----------
    fname : str
        Filename for grid file.
    Attributes
    ----------
    header : dict
        Parsed 3ds header. Relevant fields are converted to float,
        otherwise most are string values.
    signals : dict
        Dict keys correspond to channel name, with values being the
        corresponding data array.
    Raises
    ------
    UnhandledFileError
        If fname does not have a '.3ds' extension.
    """

    def __init__(self, fname=None, fromcdf=None, savecdf=0):
        print(fname)
        if fromcdf is None:
            _is_valid_file(fname, ext='3ds')
            super().__init__(fname)
            self._parse_3ds_header(self.header_raw)
            self.signals = self._load_data()
            swpp = self._derive_sweep_signal()
            
            if swpp is None:
                print(fname + " failed")
                return
            else:
                self.signals['sweep_signal'] = swpp
            self.signals['topo'] = self._extract_topo()
            if savecdf:
                self._build_xarray(savecdf=fname[:-3] + 'nc')
            else:
                self._build_xarray(savecdf=None)
            self.name = fname

            self.ncdf_name = fname[:-3] + 'nc'

            print(fname + " imported")
        else:
            self.ds = xr.open_dataset(fromcdf)
            self.ncdf_name = self.ds.cdf

    def _load_data(self):
        """
        Read binary data for Nanonis 3ds file.
        Returns
        -------
        dict
            Channel name keyed dict of 3d array.
        """
        # load grid params
        nx, ny = self.header['dim_px']
        num_sweep = self.header['num_sweep_signal']
        num_param = self.header['num_parameters']
        num_chan = self.header['num_channels']
        data_dict = dict()

        # open and seek to start of data
        f = open(self.fname, 'rb')
        f.seek(self.byte_offset)
        data_format = '>f4'
        griddata = np.fromfile(f, dtype=data_format)
        f.close()

        # pixel size in bytes
        exp_size_per_pix = num_param + num_sweep*num_chan

        # reshape from 1d to 3d

        try:
            griddata_shaped = griddata.reshape((nx, ny, exp_size_per_pix))
        except ValueError:
            ny = int(np.floor(len(griddata) / (nx * exp_size_per_pix)))
            nx = int(nx)
            exp_size_per_pix = int(exp_size_per_pix)
            griddata_shaped = griddata[:nx * ny * exp_size_per_pix].reshape((nx, ny, exp_size_per_pix))


        # experimental parameters are first num_param of every pixel
        params = griddata_shaped[:, :, :num_param]
        data_dict['params'] = params

        # extract data for each channel
        for i, chann in enumerate(self.header['channels']):
            start_ind = num_param + i * num_sweep
            stop_ind = num_param + (i+1) * num_sweep
            data_dict[chann] = griddata_shaped[:, :, start_ind:stop_ind]

        return data_dict

    def _derive_sweep_signal(self):
        """
        Computer sweep signal.
        Based on start and stop points of sweep signal in header, and
        number of sweep signal points.
        Returns
        -------
        numpy.ndarray
            1d sweep signal, should be sample bias in most cases.
        """
        # find sweep signal start and end from a given pixel value

        if 'filetype' in self.header.keys():
            
            if self.header['filetype'] == 'MLS':
                try:
                    v = []
                    mls = self.header['segments'].split(';')
                    for mj, j in enumerate(mls):
                        mv = [float(m) for m in j.split(',')]
                        vec = np.linspace(start=mv[0], stop=mv[1], num=mv[-1], dtype='float')
                        v.append(vec[:-1] if mj < len(mls) - 1 else vec)
                    self.v_raw = v #this is just to keep track of branches

                    return np.concatenate(v)

                except:
                    print('something wrong with sweep signal 1')
                    return None

            else:
                try:
                    sweep_start, sweep_end = self.signals['params'][0, 0, :2]
                    num_sweep_signal = self.header['num_sweep_signal']
                    return np.linspace(sweep_start, sweep_end, num_sweep_signal, dtype=np.float32)
                except:
                    print('something wrong with sweep signal 2')
                    return None


        else:
            try:
                print('this worked')
                sweep_start, sweep_end = self.signals['params'][0, 0, :2]
                num_sweep_signal = self.header['num_sweep_signal']
                return np.linspace(sweep_start, sweep_end, num_sweep_signal, dtype=np.float32)
            except:
                print('something wrong with sweep signal 3')
                return None

    def _extract_topo(self):
        """
        Extract topographic map based on z-controller height at each
        pixel.
        The data is already extracted, though it lives in the signals
        dict under the key 'parameters'. Currently the 4th column is the
        Z (m) information at each pixel, should update this to be more
        general in case the fixed/experimental parameters are not the
        same for other Nanonis users.
        Returns
        -------
        numpy.ndarray
            Copy of already extracted data to be more easily accessible
            in signals dict.
        """
        return self.signals['params'][:, :, 4]

    def _build_xarray(self, savecdf=None):

        ds = xr.Dataset()

        ldict = {
        'Vert. Deflection' : 'vd', 'Horiz. Deflection': 'hd', 'Input 8' : 'c', 'Current':'c',\
        'Z': 'z', 'Phase' : 'phi', 'Amplitude' : 'amp', 'Frequency Shift' : 'omega', 'OC D1 X (m)': 'liX',\
  'OC D1 Y (m)':'liY', "LockinX (V)":'liX', "LockinY (V)":'liY'}

        try:
            sweep_name = self.header['sweep_signal'].lower().split(' ')[0]
        except:
            sweep_name = 'bias'

        ds.coords[sweep_name] = self.signals['sweep_signal']
        for k,v in ldict.items():

            for m in ([i for i in self.signals.keys() if i.find(k) != -1]):
                sname = v + 'f'
                if m.find('bwd') != -1:
                    sname = v + 'r'

                ds[sname] = (('x', 'y', sweep_name), self.signals[m])

        for k,v in self.header.items():
           ds.attrs[k] = v
        ds.attrs['filename'] = self.fname

        self.ds = ds
        if savecdf is not None:
            ds.attrs['cdf'] = savecdf
            ds.to_netcdf(path=savecdf)
            pyperclip.copy(savecdf)
            print('grid(3ds)->netCDF complete! :: ' + savecdf)

    def _parse_3ds_header(self,header_raw):
        import copy
        import re
        """
        Parse raw header string.
        Empirically done based on Nanonis header structure. See Grid
        docstring or Nanonis help documentation for more details.
        Parameters
        ----------
        header_raw : str
            Raw header string from read_raw_header() method.
        Returns
        -------
        dict
            Channel name keyed dict of 3d array.
        """
        # cleanup string and remove end tag as entry
        header_entries = header_raw.split('\r\n')
        header_entries = header_entries[:-2]
        header = dict()

        header = _split_header(header_entries)

        header_dict = dict()

        # grid dimensions in pixels
        dim_px_str = _split_header_entry(header['grid_dim'])
        header_dict['dim_px'] = [int(val) for val in dim_px_str.split(' x ')]
        #
        # # filetype 'MLS' or 'Linear'
        #
        if 'filetype' in header.keys():
            header_dict['filetype'] = _split_header_entry(header['filetype'])

            if header['filetype'] == 'MLS':
                s2 = 'segment_start_(v),_segment_end_(v),_settling_(s),_integration_(s),_steps_(xn)'
                try:
                    header_dict['segments'] = header[s2]
                except:
                    print("still can't get that bias")
        else:
            header_dict['filetype'] = 'linear'

        # grid frame center position, size, angle
        if 'grid_settings' in header.keys():
            grid_str = _split_header_entry(header['grid_settings'], multiple=True)

            header_dict['pos_xy'] = [float(val) for val in grid_str[:2]]
            header_dict['size_xy'] = [float(val) for val in grid_str[2:4]]
            header_dict['angle'] = float(grid_str[-1])
        #
        # sweep signal
        # if 'sweep_signal' in header.keys():
        #     header_dict['sweep_signal'] = _split_header_entry(header['sweep_signal'])

        #
        # fixed parameters
        if 'fixed_parameters' in header.keys():
            header_dict['fixed_parameters'] = _split_header_entry(header['fixed_parameters'], multiple=True)

        #experimental parameters
        if 'experiment_parameters' in header.keys():
            header_dict['experimental_parameters'] = _split_header_entry(header['experiment_parameters'], multiple=True)

        # number of parameters (each 4 bytes)
        if '#_parameters_(4_byte)' in header.keys():
            header_dict['num_parameters'] = int(_split_header_entry(header['#_parameters_(4_byte)']))

        # experiment size in bytes
        if 'experiment_size_(bytes)' in header.keys():
            header_dict['experiment_size'] = int(_split_header_entry(header['experiment_size_(bytes)']))

        # number of points of sweep signal
        if 'points' in header.keys():
            header_dict['num_sweep_signal'] = int(_split_header_entry(header['points']))

        # channel names
        if 'channels' in header.keys():
            header_dict['channels'] = _split_header_entry(header['channels'], multiple=True)
            header_dict['num_channels'] = len(header_dict['channels'])
        #
        # measure delay
        # if 'delay_before_measuring_(s)' in header.keys():
        #     header_dict['measure_delay'] = float(_split_header_entry(header['delay_before_measuring_(s)']))

        # metadata
        # if 'experiment' in header.keys():
        #     header_dict['experiment_name'] = _split_header_entry(header['experiment'])

        # if 'start_time' in header.keys():
        #     header_dict['start_time'] = _split_header_entry(header['start_time'])
        #
        # if 'end_time' in header.keys():
        #     header_dict['end_time'] = _split_header_entry(header['end_time'])
        #
        # if 'user' in header.keys():
        #     header_dict['user'] = _split_header_entry(header['user'])
        #
        # if 'comment' in header.keys():
        #     header_dict['comment'] = _split_header_entry(header['comment'])

        # destination_keys = ['experiment_parameters','#_parameters_(4_byte)','experiment_size_(bytes)','points','delay_before_measuring_(s)','experiment','start_time','end_time','user','comment']
        header_keys = ['sweep_signal','parameters_(4_byte)','experiment_size_(bytes)','points','delay_before_measuring_(s)','experiment','start_time','end_time','user','comment']

        def check_header_flag(key_dest):
            if key_dest in header.keys():
                header_dict[key_dest] = _split_header_entry(header[key_dest])

        for kd in header_keys:
            check_header_flag(kd)


        self.header, self.header_raw2 = header_dict, header

    def _update_cdf(self, cdf=None):
        if 'cdf' in self.ds.attrs:
            savepath = self.ds.attrs['cdf']
        elif cdf != None:
            savepath = cdf
        else:
            savepath = self.fromcdf

        self.ds.to_netcdf(savepath)
        print('netCDF updated! :: ' + savepath)

    def _add_frame(self, chan='z'):
        cf = chan + 'f'
        cr = chan + 'r'
        cfr_label = chan + 'fr'
        vvec = np.append(np.flipud(self.ds.bias.values[:, None]), (self.ds.bias.values[:, None]))
        cfr = np.append(np.flip(self.ds[cf].values, axis=2), self.ds[cr].values, axis=2)
        x, y, z = cfr.shape
        cfrxr = xr.DataArray(cfr, coords=[np.arange(x), np.arange(y), vvec], dims=['x', 'y', 'biasfr'])
        self.ds.__setitem__(cfr_label, cfrxr)





class Scan(NanonisFile):

    """
    Nanonis scan file class.
    Contains data loading methods specific to Nanonis sxm files. The
    header is terminated by a 'SCANIT_END' tag followed by the \1A\04
    code. The NanonisFile header parse method doesn't account for this
    so the Scan __init__ method just adds 4 bytes to the byte_offset
    attribute so as to not include this as a datapoint.
    Data is structured a little differently from grid files, obviously.
    For each pixel in the scan, each channel is recorded forwards and
    backwards one after the other.
    Currently cannot take scans that do not have both directions
    recorded for each channel, nor incomplete scans.
    Parameters
    ----------
    fname : str
        Filename for scan file.
    Attributes
    ----------
    header : dict
        Parsed sxm header. Some fields are converted to float,
        otherwise most are string values.
    signals : dict
        Dict keys correspond to channel name, values correspond to
        another dict whose keys are simply forward and backward arrays
        for the scan image.
    Raises
    ------
    UnhandledFileError
        If fname does not have a '.sxm' extension.
    """

    def __init__(self, fname):
        _is_valid_file(fname, ext='sxm')
        super().__init__(fname)
        self.header = self._parse_sxm_header(self.header_raw)
        # data begins with 4 byte code, add 4 bytes to offset instead
        self.byte_offset += 4

        # load data
        #return self.header_raw
        self.signals = self._load_data()
        self._to_dataset()

    def _load_data(self):
        """
        Read binary data for Nanonis sxm file.
        Returns
        -------
        dict
            Channel name keyed dict of each channel array.
        """
        channs = list(self.header['data_info']['Name'])
        nchanns = len(channs)
        nx, ny = self.header['scan_pixels']

        # assume both directions for now
        ndir = 2

        data_dict = dict()

        # open and seek to start of data
        f = open(self.fname, 'rb')
        f.seek(self.byte_offset)
        data_format = '>f4'
        scandata = np.fromfile(f, dtype=data_format)
        f.close()

        # reshape
        scandata_shaped = scandata.reshape(nchanns, ndir, nx, ny)

        # extract data for each channel
        for i, chann in enumerate(channs):
            chann_dict = dict(forward=np.flipud(scandata_shaped[i, 0, :, :]),
                              backward=np.flipud(scandata_shaped[i, 1, :, :]))
            data_dict[chann] = chann_dict

        return data_dict


    def _to_dataset(self):
        import xarray as xr
        ds = xr.Dataset()

        ldict = {
            'Vert. Deflection': 'vd', 'Horiz. Deflection': 'hd', 'Input 8': 'c', 'Current': 'c', \
            'Z': 'z', 'Phase': 'phi', 'Amplitude': 'amp', 'Frequency Shift': 'omega'}


        x = np.arange(self.header['scan_pixels'][0])*self.header['scan_range'][0]
        y = np.arange(self.header['scan_pixels'][1])*self.header['scan_range'][1]

        ds.coords['x'] = x
        ds.coords['y'] = y


        for k, v in ldict.items():

            for m in ([i for i in self.signals.keys() if i.find(k) != -1]):

                dat = self.signals[m]
                if 'forward' in dat.keys():
                    sname = v + 'f'
                    ds[sname] = (('x', 'y'), dat['forward'])

                if 'backward' in dat.keys():
                    sname = v + 'b'
                    ds[sname] = (('x', 'y'), dat['backward'])

        ds.attrs = self.header
        self.ds = ds

    def _parse_sxm_header(self, header_raw):
        """
        Parse raw header string.
        Empirically done based on Nanonis header structure. See Scan
        docstring or Nanonis help documentation for more details.
        Parameters
        ----------
        header_raw : str
            Raw header string from read_raw_header() method.
        Returns
        -------
        dict
            Channel name keyed dict of each channel array.
        """

        header_dict = dict()
        header_entries = header_raw.split('\n')

        h_names = header_entries[0::2]

        h_names = [j for j in header_entries if j.isupper() and j.startswith(':') and j.endswith(':')]
        h_vals = [j for j in header_entries if j not in h_names]

        for j in h_names:
            sb,sa = header_raw.strip().split(j)
            value = sa.strip().split(':')[0].strip()
            header_dict[j.lower().strip(':')] = value

        entries_to_be_split = ['scan_offset',
                               'scan_pixels',
                               'scan_range',
                               'scan_time']

        entries_to_be_floated = ['scan_offset',
                                 'scan_range',
                                 'scan_time',
                                 'bias',
                                 'acq_time']

        entries_to_be_inted = ['scan_pixels']
        entries_to_be_tabled = ['data_info', 'z-controller']

        for k in entries_to_be_split:
            header_dict[k] = header_dict[k].split()

        for k in entries_to_be_floated:
            if isinstance(header_dict[k], list):
                header_dict[k] = np.asarray(header_dict[k], dtype=np.float)
            else:
                header_dict[k] = np.float(header_dict[k])

        for k in entries_to_be_inted:
             header_dict[k] = np.asarray(header_dict[k], dtype=np.int)

        for k in entries_to_be_tabled:
            s2 = header_dict[k].split('\n')
            s3 = [j.strip().split('\t') for j in s2]
            header_dict[k]= pd.DataFrame(s3[1:], columns=s3[0])

        header_dict['fname'] = self.fname
        return header_dict

class Spectrum(NanonisFile):

    """
    Nanonis point spectroscopy file class.
    These files are a little easier to handle since they are stored in
    ascii format.
    Parameters
    ----------
    fname : str
        Filename for spec file.
    Attributes
    ----------
    header : dict
        Parsed dat header.
    Raises
    ------
    UnhandledFileError
        If fname does not have a '.dat' extension.
    """

    def __init__(self, fname):
        _is_valid_file(fname, ext='dat')
        super().__init__(fname)
        self.header = self._parse_dat_header(self.header_raw)
        self.signals = self._load_data()

    def _load_data(self):
        """
        Loads ascii formatted .dat file.
        Header ended by '[DATA]' tag.
        Returns
        -------
        dict
            Keys correspond to each channel recorded, including
            saved/filtered versions of other channels.
        """

        # done differently since data is ascii, not binary
        f = open(self.fname, 'r')
        f.seek(self.byte_offset)
        data_dict = dict()

        column_names = f.readline().strip('\n').split('\t')
        f.close()
        
        header_lines = len(self.header) + 4
        specdata = np.genfromtxt(self.fname, delimiter='\t', skip_header=header_lines)

        for i, name in enumerate(column_names):
            data_dict[name] = specdata[:, i]

        return data_dict

    def _parse_dat_header(self, header_raw):
        """
        Parse point spectroscopy header.
        Each key-value pair is separated by '\t' characters. Values may be
        further delimited by more '\t' characters.
        Returns
        -------
        dict
            Parsed point spectroscopy header.
        """
        header_entries = header_raw.split('\r\n')
        header_entries = header_entries[:-3]
        header_dict = dict()
        for entry in header_entries:
            key, val, _ = entry.split('\t')
            header_dict[key] = val

        return header_dict

    def _build_xarray(self, savecdf=None):
        import xarray as xr
        ds = xr.Dataset()

        ldict = {
            'Vert. Deflection': 'vd', 'Horiz. Deflection': 'hd', 'Input 8': 'c', 'Current': 'c', \
            'Z': 'z', 'Phase': 'phi', 'Amplitude': 'amp', 'Frequency Shift': 'omega'}

        try:
            #sweep_name = self.header['sweep_signal'].lower().split(' ')[0]
            sweep_name = self.header['Experiment'].lower().split(' ')[0]

        except:
            sweep_name = 'bias'

        if sweep_name == 'bias':
            vf_name = 'Bias'
            vr_name = 'Bias [bwd]'

        ds.coords[sweep_name] = self.signals['Bias (V)']
        for k, v in ldict.items():

            for m in ([i for i in self.signals.keys() if i.find(k) != -1]):
                sname = v + 'f'
                if m.find('bwd') != -1:
                    sname = v + 'r'

                ds[sname] = (('x', 'y', sweep_name), self.signals[m])

        for k, v in self.header.items():
            ds.attrs[k] = v
        ds.attrs['filename'] = self.fname

        self.ds = ds




class UnhandledFileError(Exception):

    """
    To be raised when unknown file extension is passed.
    """
    pass


class FileHeaderNotFoundError(Exception):

    """
    To be raised when no header information could be determined.
    """
    pass


def _clean_sxm_header(header_dict):
    """
    Cleanup header dicitonary key-value pairs.
    Parameters
    ----------
    header_dict : dict
        Should be dict returned from _parse_sxm_header method.
    Returns
    -------
    clean_header_dict : dict
        Cleaned header dictionary.
    """
    pass


def _split_header_entry(entry, multiple=False):
    """
    Split 3ds header entries by '=' character. If multiple values split
    those by ';' character.
    """

    #_, val_str = entry.split("=")

    if multiple:
        return entry.strip('"').split(';')
    else:
        return entry.strip('"')

def _split_header(h, multiple=False):
    header = dict()
    for j in h:
        key, val = j.split('=',1)
        header['_'.join(key.lower().split(' '))] = val
    return header

def save_array(file, arr, allow_pickle=True):
    """
    Wrapper to numpy.save method for arrays.
    The idea would be to use this to save a processed array for later
    use in a matplotlib figure generation scripts. See numpy.save
    documentation for details.
    Parameters
    ----------
    file : file or str
        File or filename to which the data is saved.  If file is a file-
        object, then the filename is unchanged.  If file is a string, a
        ``.npy`` extension will be appended to the file name if it does
        not already have one.
    arr : array_like
        Array data to be saved.
    allow_pickle : bool, optional
        Allow saving object arrays using Python pickles. Reasons for
        disallowing pickles include security (loading pickled data can
        execute arbitrary code) and portability (pickled objects may not
        be loadable on different Python installations, for example if
        the stored objects require libraries that are not available, and
        not all pickled data is compatible between Python 2 and Python
        3). Default: True
    """
    np.save(file, arr, allow_pickle=allow_pickle)

def load_array(file, allow_pickle=True):
    """
    Wrapper to numpy.load method for binary files.
    See numpy.load documentation for more details.
    Parameters
    ----------
    file : file or str
        The file to read. File-like objects must support the
    ``seek()`` and ``read()`` methods. Pickled files require that the
    file-like object support the ``readline()`` method as well.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons
        for disallowing pickles include security, as loading pickled
        data can execute arbitrary code. If pickles are disallowed,
        loading object arrays will fail. Default: True
    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned
        instance of NpzFile class must be closed to avoid leaking file
        descriptors.
    """
    return np.load(file)


def _parse_scan_header_table(table_list):
    """
    Parse scan file header entries whose values are tab-separated
    tables.
    """
    table_processed = []
    for row in table_list:
        # strip leading \t, split by \t
        table_processed.append(row.strip('\t').split('\t'))

    # column names are first row
    keys = table_processed[0]
    values = table_processed[1:]

    zip_vals = zip(*values)

    return dict(zip(keys, zip_vals))


def _is_valid_file(fname, ext):
    """
    Detect if invalid file is being initialized by class.
    """
    if fname[-3:] != ext:
        raise UnhandledFileError('{} is not a {} file'.format(fname, ext))
