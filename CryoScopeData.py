import numpy as np
import h5py

from UsefulUtils import Bunch
import UsefulUtils as uu
from matplotlib import pyplot as plt

def load_hydraharp_histogram(filename, **kwargs):
    h5f = h5py.File(filename, 'r')

    s = Bunch(**kwargs)
    
    s.file = uu.get_file_title(filename)
    
    s.times = np.array(h5f['measurement']['hydraharp_histogram']['time_array'])
    s.ints = np.array(h5f['measurement']['hydraharp_histogram']['time_histogram'])
    s.x_pos = h5f['hardware']['ni_xy_voltage_scanner']['settings'].attrs['x_position']
    s.y_pos = h5f['hardware']['ni_xy_voltage_scanner']['settings'].attrs['y_position']
    
    s.intg_time =  h5f['hardware']['hydraharp']['settings'].attrs['Tacq']
    s.spec_cwl = h5f['hardware']['iHR320_spectrometer']['settings'].attrs['wavelength']
    
    h5f.close()
    
    return s

def load_APD_2D_scan(filename):
    h5f = h5py.File(filename, 'r')
    
    apd_scan = Bunch()

    apd_scan.file = uu.get_file_title(filename)
    
    apd_scan.int_map = h5f['measurement']['APD_2D_scan']['count_rate_map'][0]
    apd_scan.extents = np.array(h5f['measurement']['APD_2D_scan']['imshow_extent'])    
    
    h5f.close()
    
    return apd_scan   
	
def load_andor_ccd_readout(filename):
    h5f = h5py.File(filename, 'r')
    
    s = Bunch()
    
    s.file = uu.get_file_title(filename)
    
    s.wls = np.array(h5f['measurement']['andor_ccd_readout']['wls'])
    s.ints = np.array(h5f['measurement']['andor_ccd_readout']['spectrum'][0])
    s.x_pos = h5f['hardware']['ni_xy_voltage_scanner']['settings'].attrs['x_position']
    s.y_pos = h5f['hardware']['ni_xy_voltage_scanner']['settings'].attrs['y_position']
    s.temperature = h5f['hardware']['LakeShore331']['settings'].attrs['temperature']
    s.spec_cwl = h5f['hardware']['iHR320_spectrometer/settings'].attrs['wavelength']
    s.intg_time =  h5f['hardware']['andor_ccd']['settings'].attrs['exposure_time']

    s.xaxis_calib = h5f['measurement']['andor_ccd_readout']['settings'].attrs['wl_calib']

    s.grating = h5f['hardware']['iHR320_spectrometer/settings'].attrs['grating']
    s.grating_cal = np.array(h5f['hardware']['iHR320_spectrometer/settings'].attrs['grating_calibrations'][s.grating])
    
    h5f.close()
    
    return s
    
def load_step_and_glue(filename, description=""):
    h5_f = h5py.File(filename, 'r')
    
    o = Bunch(filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description
              )
    o.x_pos = h5_f['hardware/ni_xy_voltage_scanner/settings'].attrs['x_position']
    o.y_pos = h5_f['hardware/ni_xy_voltage_scanner/settings'].attrs['y_position']
    o.exp_time = h5_f['hardware/andor_ccd/settings'].attrs['exposure_time']
    
   
    o.grating_positions = np.array(h5_f['measurement/step_and_glue/grating_wavelengths'])
    o.wls_raw = h5_f['measurement/step_and_glue/spectra_wavelengths'][()].copy()
    o.ints_raw = h5_f['measurement/step_and_glue/sprecta_intensities'][()].copy()
    
    wls, ints = merge_step_and_glue_specs_v2(o.grating_positions, o.wls_raw, o.ints_raw)
    
    o.wls = wls
    o.ints = ints
    
    return o

def merge_step_and_glue_specs_v2(centers, wavelengths, specs):
    # Initialize with the first spectrum
    wls = wavelengths[0][:]
    raw_ints = specs[0][:]

    # Progressively merge spectra into the lists of wls.
    for i in range(1, centers.shape[0]):
        # Determine the array indices and size of the regions of overlap
        i_overlap_start = np.searchsorted(wls, wavelengths[i][0])
        overlap_size_1 = wls.shape[0] - i_overlap_start
        overlap_size_2 = np.searchsorted(wavelengths[i], wls[-1])
        #print(i_overlap_start, overlap_size_1, overlap_size_2)

        # Generate a uniformly spaced points in the overlap region 
        wls_overlap = np.linspace(wavelengths[i][0], wls[-1], 
                                  int((overlap_size_1+overlap_size_2)/2))

        # Take the average of the interpolated values from the two data sets in the
        # interpolated region.
        raw_int_overlap = (np.interp(wls_overlap, wls[i_overlap_start:], raw_ints[i_overlap_start:] ) + 
                           np.interp(wls_overlap, wavelengths[i][0:overlap_size_2], 
                                     specs[i][0:overlap_size_2]))/2.0

        # Append the averaged overlapped data to the non-overlapped data at the beginning.
        wls = np.append(wls[0:i_overlap_start], wls_overlap)
        raw_ints = np.append(raw_ints[0:i_overlap_start], raw_int_overlap)

        # Append the remaining non-overlapped data 
        wls = np.append(wls, wavelengths[i][overlap_size_2:])
        raw_ints = np.append(raw_ints, specs[i][overlap_size_2:])

    return wls, raw_ints

def load_rotation_mount_step_and_glue(filename, description = ""):
    
    h5_f = h5py.File(filename, 'r')
    
    o = Bunch(filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description
              )
    o.x_pos = h5_f['hardware/ni_xy_voltage_scanner/settings'].attrs['x_position']
    o.y_pos = h5_f['hardware/ni_xy_voltage_scanner/settings'].attrs['y_position']
    o.exp_time = h5_f['hardware/andor_ccd/settings'].attrs['exposure_time']
    
   
    o.grating_positions = np.array(h5_f['measurement/rotation_mount_step_and_glue/grating_wavelengths'])
    o.rotation_angles = np.array(h5_f['measurement/rotation_mount_step_and_glue/rotation_angles'])
    
    o.wls = [] # = np.zeros(len(o.rotation_angles))
    o.ints = []
    
    for i in range(0, len(h5_f['measurement/rotation_mount_step_and_glue/spectra_wavelengths'])):
        
        o.wls_raw = h5_f['measurement/rotation_mount_step_and_glue/spectra_wavelengths'][i][()].copy()
        o.ints_raw = h5_f['measurement/rotation_mount_step_and_glue/spectra_intensities'][i][()].copy()
        
        wls, ints = merge_step_and_glue_specs_v2(o.grating_positions, o.wls_raw, o.ints_raw)  
        
        o.wls.append(wls)
        o.ints.append(ints)

    o.evs = np.flip(1240/np.array(o.wls))
    o.specs_evs = np.flip(np.array(o.ints), axis=1)

    o.even_evs = np.linspace(np.min(o.evs), np.max(o.evs), o.evs.shape[1])
    o.even_specs_evs = np.zeros((len(o.ints), o.even_evs.size))
    for spec_no in range(o.specs_evs.shape[0]):
        o.even_specs_evs[spec_no, :] = np.interp(o.even_evs, o.evs[spec_no, :], o.specs_evs[spec_no,:])
    
    return o

def load_andor_hyperspec_scan(filename, description=""):
    h5_f = h5py.File(filename, 'r')
    
    o = Bunch(filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description
              )
  
    
    o.exp_time = h5_f['hardware/andor_ccd/settings'].attrs['exposure_time']
    o.wls = np.array(h5_f['measurement/andor_hyperspec_scan/wls'])
    o.specmap = np.array(h5_f['measurement/andor_hyperspec_scan/spec_map'])
    o.specmap1 = np.array(h5_f['measurement/andor_hyperspec_scan/spec_map'])[0,:]
    o.h_pos = np.array(h5_f['measurement/andor_hyperspec_scan/scan_h_positions'])
    o.v_pos = np.array(h5_f['measurement/andor_hyperspec_scan/scan_v_positions'])
    o.extent = np.array(h5_f['measurement/andor_hyperspec_scan/imshow_extent'])

    o.rows = o.specmap.shape[1]
    o.cols = o.specmap.shape[2]
    
    # Calculate a default intensity map
    o.int_map = np.sum(o.specmap[0], axis=2)

    h5_f.close()
    
    return o

def load_ccd_time_series(filename, description = ""):
    h5_f = h5py.File(filename, 'r')
    
    o = Bunch(filename = filename,
              filetitle = uu.get_file_title(filename),
              description = description
              )
  
    
    o.exp_time = h5_f['hardware/andor_ccd/settings'].attrs['exposure_time']
    o.wls = np.array(h5_f['measurement/ccd_time_series/wls_array'][0])
    o.specs = np.array(h5_f['measurement/ccd_time_series/spec_array'])
    o.evs = 1240/np.flip(o.wls)
    o.specs_evs = np.flip(o.specs, axis = 1) #neglects the proper jacobian transformation
    o.times = np.array(h5_f['measurement/ccd_time_series/times'])
    o.spec_cwl = h5_f['hardware/iHR320_spectrometer/settings'].attrs['wavelength']
    o.imshow_extent_wls = [o.wls[0],o.wls[-1], 0,o.times[-1] - o.times[0]] #sets the first acquistion time as reference
                                                                             #use origin = 'lower' 
    o.imshow_extent_evs = [o.evs[0],o.evs[-1], 0,o.times[-1] - o.times[0]]
    #this is a change!

    #interpolate spectra on evenly space x-axis
    o.even_evs = np.linspace(np.min(o.evs), np.max(o.evs), o.evs.size)
    o.even_specs_evs = np.zeros_like(o.specs_evs)
    for spec_no in range(o.specs_evs.shape[0]):
        o.even_specs_evs[spec_no, :] = np.interp(o.even_evs, o.evs, o.specs_evs[spec_no,:])
    
    o.x_pos = h5_f['hardware/ni_xy_voltage_scanner/settings'].attrs['x_position']
    o.y_pos = h5_f['hardware/ni_xy_voltage_scanner/settings'].attrs['y_position']

    h5_f.close()
    
    return o
    
def load_grumpstrup_map_data(dat_file_path, description = None, **kwargs):
    o = Bunch(filename = dat_file_path.split('.dat')[0],
              description = description,
              **kwargs)
    o.data = np.genfromtxt(dat_file_path, delimiter = '\t')
    o.map = np.reshape(o.data[:,2], (len(set(o.data[:,0])), len(set(o.data[:,1] ))), order = 'C')
    o.xx = np.reshape(o.data[:,0], (len(set(o.data[:,0])), len(set(o.data[:,1] ))), order = 'C')
    o.yy = np.reshape(o.data[:,1], (len(set(o.data[:,0])), len(set(o.data[:,1] ))), order = 'C')
    o.extent = np.min(o.xx), np.max(o.xx), np.min(o.yy), np.max(o.yy)
    o.map[1::2] = np.flip(o.map[1::2], axis = 1) 
    return o

def load_grumpstrup_kin_data(dat_file_path, description = None, **kwargs):
    o = Bunch(filename = dat_file_path.split('.dat')[0],
              description = description,
              **kwargs)
    o.data = np.genfromtxt(dat_file_path, delimiter = '\t')
    o.times = o.data[:,0]
    o.trace = o.data[:,1]
    return o
    
def load_diff_ref(filename):
    o = Bunch()
    with h5py.File(filename) as h5:
        o.wls = np.array(h5['measurement']['diff_reflectance']['merged_wls'])
        o.specs = np.array(h5['measurement']['diff_reflectance']['diff_reflectance'])
        
        o.specs_pos0 = np.array(h5['measurement']['diff_reflectance']['sprecta_intensities_pos0'])
        o.specs_pos1 = np.array(h5['measurement']['diff_reflectance']['sprecta_intensities_pos1'])
        
        o.wls_pos0 = np.array(h5['measurement']['diff_reflectance']['spectra_wavelengths_pos0'])
        o.wls_pos1 = np.array(h5['measurement']['diff_reflectance']['spectra_wavelengths_pos1'])
        
    return o
 

	
def import_h5_to_dict(h5_file_path):
	"""
	Loads an .h5 file generated by ScopeFoundry to a dictionary.
	Works for most .h5 data files generated
	does not work for composite data files, e.g. powerSeries data
	"""
	try:
		h5 = h5py.File(h5_file_path)
	except:
		print('Failed to open .h5 file at the location: {}'.format(h5_file_path))
		return None
	try:
		export_dict = dict()
		export_dict['hardware'] = dict()
		export_dict['measurement'] = dict()
		export_dict['analysis'] = dict()
		
		for hardware_comp in h5['hardware']:
			export_dict['hardware']['{}'.format(hardware_comp)] = dict()
			#print('{}'.format(hardware_comp))
			for attr in h5['hardware']['{}'.format(hardware_comp)]['settings'].attrs:
				export_dict['hardware']['{}'.format(hardware_comp)]['{}'.format(attr)] = \
				h5['hardware']['{}'.format(hardware_comp)]['settings'].attrs[attr]
				
		
		for measurement in list(h5['measurement'].keys()):
			export_dict['measurement'][measurement] = dict()
			for group in h5['measurement'][measurement]: 
				if group == 'settings':
					export_dict['measurement'][measurement][group] = dict()
					for group1 in h5['measurement'][measurement][group]:
						export_dict['measurement'][measurement][group][group1] = dict()
						for attr in h5['measurement'][measurement][group][group1].attrs:
							export_dict['measurement'][measurement][group][group1][attr] = \
							h5['measurement'][measurement][group][group1].attrs[attr]
					for attr in h5['measurement'][measurement][group].attrs:
						export_dict['measurement'][measurement][group][attr] = h5['measurement'][measurement][group].attrs[attr]
				else:
					export_dict['measurement'][measurement][group] = np.array(h5['measurement'][measurement][group])
	except:
		print('Failed to properly convert .h5 sturcutre to dict')
		export_dict = none
	finally:
		h5.close()
		return export_dict


## Plotter functions for making profiles across hyperspectral images
import scipy.ndimage

def get_linecut(data, start_row, start_col, end_row, end_col, 
                over_sample_factor=1, avg_width=0, avg_step=0.25,
                interp_order = 1):
    
        
    
    #Length of linecut in pixel units
    linecut_len = np.hypot(end_col - start_col, end_row - start_row)
    
    #Row and column coordinates for the linecut
    sample_count = int(linecut_len*over_sample_factor)
    row_coords = np.linspace(start_row, end_row, sample_count)
    col_coords = np.linspace(start_col, end_col, sample_count)  
    
    starts = []
    ends = []
    
    if avg_width/avg_step <2:
        #Cut across a single line;  do not average in orthogonal direction    
        starts.append((start_row, start_col))
        ends.append((end_row, end_col))
        
        line_int = scipy.ndimage.map_coordinates(data, np.vstack((row_coords, col_coords)), order=interp_order)
    else:
        # Average in an orthogonal direction by the average_width and avg_width_step parameters
                
        # Calculate the number of averages to take
        steps = int(avg_width/avg_step)
        
        delta_x = avg_step*(end_col-start_col)/linecut_len    #d*sin(alpha)
        delta_y = avg_step*(end_row-start_row)/linecut_len    #d*cos(alpha)
        
        line_int = np.zeros(sample_count)
        
        count = 0
        
        for i in range(-1*int(steps/2), int(steps/2)+1, 1):
            #print(i, row1-i*delta_x, col1+i*delta_y)
            
            start_row_tmp = start_row-i*delta_x
            end_row_tmp   = end_row-i*delta_x
            start_col_tmp = start_col+i*delta_y
            end_col_tmp   = end_col+i*delta_y
            
            rows_tmp = np.linspace(start_row_tmp, end_row_tmp, sample_count)
            cols_tmp = np.linspace(start_col_tmp, end_col_tmp, sample_count)

            line_int += scipy.ndimage.map_coordinates(data, np.vstack((rows_tmp,cols_tmp)), order=interp_order) 
            
            starts.append((start_row_tmp, start_col_tmp))
            ends.append((end_row_tmp, end_col_tmp))
            count += 1
        
        # Compute the average
        line_int = line_int/count
       
        
    return line_int, row_coords, col_coords, starts, ends


def plot_cross_section_spec_for_hyperspec(hs, start_x, start_y, end_x, end_y, avg_width_in):
      
    # do some calculations on the hyperspectral object; this approach is not the best, because it modifies hs a bit.  
    hs.extent_width = np.abs(hs.extent[1] - hs.extent[0])
    hs.extent_height = np.abs(hs.extent[3] - hs.extent[2])
    hs.spec_pixels = hs.specmap.shape[-1]
    hs.width = hs.extent_width 
    hs.height = hs.extent_height
    hs.px_w = hs.width/hs.cols
    hs.px_h = hs.height/hs.rows
    hs.pixel_size = (hs.px_w+hs.px_h)/2
        
    img = hs
    
    # Convert x and y coordinates into rows and columns
    start_row = img.rows*(start_y - img.extent[2])/img.extent_height
    end_row   = img.rows*(end_y - img.extent[2])/img.extent_height
    
    start_col = img.cols * ((start_x - img.extent[0])/img.extent_width)
    end_col   = img.cols * ((end_x - img.extent[0])/img.extent_width)
    
    # Get the linecut
    line_int, rows, cols, starts, ends = get_linecut(img.int_map, start_row, start_col, end_row, end_col,
                                                     avg_width=avg_width_in, over_sample_factor=1.5)
    specs = np.empty((line_int.shape[0],img.spec_pixels))
    
    for i in range(img.spec_pixels): #in img.hs_map[rows,cols,start]
        line_int, rows, cols, starts, ends = get_linecut(img.specmap[0, :,:,i], start_row, start_col, end_row, end_col,
                                                     avg_width=avg_width_in, over_sample_factor=1.5)
        specs[:,-i] = line_int
        
    # Convert position on linecut back to relative spatial dimensions
    position = np.sqrt(((cols-cols[0])*img.pixel_size)**2 + 
                       ((rows-rows[0])*img.pixel_size)**2)
    
    
    #for j in range(specs.shape[0]):
    #    specs[j,:]= uu.norm(specs[j,:])
    
    plt.figure(figsize=(7,3))
    plt.subplot(1, 2, 1)
    plt.imshow(img.int_map, extent=img.extent, origin='lower')
    #plt.colorbar(shrink=0.5)
    
    for s,e in zip(starts, ends):
        sx = img.extent[0] + s[1]*img.pixel_size
        sy = img.extent[2] + s[0]*img.pixel_size
        
        ex = img.extent[0] + e[1]*img.pixel_size
        ey = img.extent[2] + e[0]*img.pixel_size
        
        plt.plot([sx, ex], [sy, ey], '-', lw=0.3, color='w')
         
    plt.subplot(1, 2, 2)
    
    extent_spec = (img.wls[0], img.wls[-1], position[0], position[-1])
    plt.imshow(specs, aspect = 'auto', extent = extent_spec,origin='lower',interpolation='none')
    cb = plt.colorbar(shrink=0.9)
    cb.set_label('Intensity (cts)', rotation=-90, labelpad=10)
    
    plt.clim(0,300)
    #plt.xlim(112,550)
    #plt.axvline(410)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('linecut position (µm)')
    plt.suptitle('({0:.1f},{1:.1f}) -> ({2:.1f},{3:.1f})'.format(start_x,start_y,end_x,end_y), fontsize=10)

    plt.tight_layout()

def convert_ccd_px_2_wl(px_index, f, delta, gamma, n0, offset_adjust, d_grating, x_pixel,  wl_center, m_order=1, curvature=0, binning=1):
    def wl_p_calib(px, n0, offset_adjust, wl_center, m_order, d_grating, x_pixel, f, delta, gamma, curvature=0):
        #print('wl_p_calib:', px, n0, offset_adjust, wl_center, m_order, d_grating, x_pixel, f, delta, gamma, curvature)
        #consts
        #d_grating = 1./150. #mm
        #x_pixel   = 16e-3 # mm
        #m_order   = 1 # diffraction order, unitless
        n = px - (n0+0*wl_center)

        #print('psi top', m_order* wl_center)
        #print('psi bottom', (2*d_grating*np.cos(gamma/2)) )

        psi = np.arcsin( m_order* wl_center / (2*d_grating*np.cos(gamma/2)))
        eta = np.arctan(n*x_pixel*np.cos(delta) / (f+n*x_pixel*np.sin(delta)))

        return ((d_grating/m_order)
                        *(np.sin(psi-0.5*gamma)
                          + np.sin(psi+0.5*gamma+eta))) + curvature*n**2

    binned_px = binning*px_index + 0.5*(binning-1)
    wl = wl_p_calib(binned_px, n0, offset_adjust, wl_center, m_order, d_grating, x_pixel, f, delta, gamma, curvature) + offset_adjust
          
    return wl
        
    
