import os
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as pl
import matplotlib.colors as mc
import matplotlib.ticker as ticker
import numpy as np
from numpy import uint32, uint64, float64, float32
from numpy import sqrt, log, exp, arccosh, sinh, cosh, tanh, pi, cos, sin, arctan2, arccos
from scipy import ndimage
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

def Calc_LambdaJeans(rho_cgs,cs_kms):
    cs_cgs = cs_kms*1e5
    G = 6.67e-8 # cm^3/g/s^2
    lambdaj = sqrt(pi/G/rho_cgs)*cs_cgs
    return lambdaj # returns lambdaj in cm

def Calc_MassJeans(rho_cgs,cs_kms):
    lambdaj = Calc_LambdaJeans(rho_cgs,cs_kms)
    Mjeans = 4/3*pi*rho_cgs*(lambdaj/2e33)*lambdaj*lambdaj
    #Mjeans = Mjeans / 2e33 # convert from g to Msun
    return Mjeans

def Calc_cs(T,mu=2.0):
    mp = 1.6726231e-24 # proton mass in g
    kb = 1.3806485e-16 # Boltzmann constant in g*cm^2/s^2/K 
    cs_cgs = sqrt((5.0/3.0)*kb*T/(mu*mp))
    cs_kms = cs_cgs/1e5
    return cs_kms

def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    if (int(a)==1):
        return r'$10^{{{}}}$'.format(b)
    else:
        return r'${} \times 10^{{{}}}$'.format(a, b)

header_names = ('num_particles', 'mass', 'time', 'redshift', 'flag_sfr', 'flag_feedback', 'num_particles_total', 'flag_cooling', \
                'num_files', 'boxsize', 'omega0', 'omegaLambda', 'hubble0', 'flag_stellarage', 'flag_metals', \
                'npartTotalHighWord', 'flag_entropy_instead_u', 'flag_doubleprecision', \
                'flag_lpt_ics', 'lpt_scalingfactor', 'flag_tracer_field', 'composition_vector_length', 'buffer')

header_sizes = ((uint32, 6), (float64, 6), (float64, 1), (float64, 1), (uint32, 1), (uint32, 1), \
                (uint32, 6), (uint32, 1), (uint32, 1), (float64, 1), (float64, 1), (float64, 1), \
                (float64, 1), (uint32, 1), (uint32, 1), (uint32, 6), (uint32, 1), (uint32, 1), \
                (uint32, 1), (float32, 1), (uint32, 1), (uint32, 1), (np.uint8, 40))

def read_header(f):
    """ Read the binary header file into a dictionary """
    block_size = np.fromfile(f, uint32, 1)[0]
    header = dict(((name, np.fromfile(f, dtype=size[0], count=size[1])) \
                 for name, size in zip(header_names, header_sizes)))
    assert(np.fromfile(f, uint32, 1)[0] == 256)
    return header

def readu(f, dtype=None, components=1):
    """ Read a numpy array from the unformatted fortran file f """
    data_size = np.fromfile(f, uint32, 1)[0]
    count = int(data_size/np.dtype(dtype).itemsize)
    arr = np.fromfile(f, dtype, count)
    final_block = np.fromfile(f, uint32, 1)[0]
    # check the flag at the beginning corresponds to that at the end
    assert(data_size == final_block)
    return arr

def readIDs(f, count):
    """ Read a the ID block from a binary snapshot file """
    data_size = np.fromfile(f, uint32, 1)[0]
    f.seek(-4, 1)
    count = int(count)
    if data_size / 4 == count: dtype = uint32
    elif data_size / 8 == count: dtype = uint64
    else: raise Exception('Incorrect number of IDs requested')
    print("ID type: ", dtype)
    return readu(f, dtype, 1)

def read_snapshot_file(filename):
    """ Reads a binary arepo file """
    f = open(filename, mode='rb')
    print("Loading file %s" % filename)
    data = {} # dictionary to hold data
    data_gas = {}
    data_sink = {}
    # read the header
    header = read_header(f)
    nparts = header['num_particles']
    masses = header['mass']
    total = nparts.sum()
    n_gas = nparts[0]
    n_sink = nparts[5]
    print('Particles', nparts)
    print('Gas particles', n_gas)
    print('Sink particles', n_sink)
    print('Time = ', header['time'])
    precision = float32
    print('Reading positions')
    data['pos'] = readu(f, precision, 3).reshape((total, 3))
    print('Reading velocities')
    data['vel'] = readu(f, precision, 3).reshape((total, 3))
    print('Reading IDs')
    data['id'] = readIDs(f, total)
    print('Reading masses')
    data['mass'] = readu(f, precision, 1)
    print('Reading internal energy')
    data['u_therm'] = readu(f, precision, 1)
    print('Reading densities')
    data['rho'] = readu(f, precision, 1)
    print('Reading chemical abundances')
    data['chem'] = readu(f, precision, 3).reshape((n_gas, 3))
    print('Reading dust temperatures')
    data['tdust'] = readu(f, precision, 1)
    f.close()
    for field in data:
        data_gas[field] = data[field][0:n_gas]
    data_sink['pos'] = data['pos'][n_gas:(n_gas+n_sink)]
    data_sink['vel'] = data['vel'][n_gas:(n_gas+n_sink)]
    data_sink['id'] = data['id'][n_gas:(n_gas+n_sink)]
    data_sink['mass'] = data['mass'][n_gas:(n_gas+n_sink)]
    return data_gas, data_sink, header

def read_arepo_image(filename):
    f = open(filename, mode='rb')
    print("Loading file %s" % filename)

    npix_x = np.fromfile(f, uint32, 1)
    npix_y = np.fromfile(f, uint32, 1)

    print(npix_x, npix_y)
    arepo_image = np.fromfile(f, float32, int(npix_x*npix_y)).reshape((int(npix_x), int(npix_y)))
    arepo_image = np.rot90(arepo_image)
    f.close()
    return arepo_image

def read_projections(fname, isnap):
    Sigma = read_arepo_image(fname+'/density_proj_%03d'%isnap)
    xHp =   read_arepo_image(fname+'/xHP_proj_%03d'%isnap)
    xH2 =   read_arepo_image(fname+'/xH2_proj_%03d'%isnap)
    xCO =   read_arepo_image(fname+'/xCO_proj_%03d'%isnap)

    xHe = 0.1
    mp = 1.6726231e-24
    kb = 1.3806485e-16
    xHI = 1 - xHp -2*xH2
    NHtot = (Sigma * arepoColumnDensity) / ((1. + 4.0 * xHe) * mp)
    NHp = xHp*NHtot
    NH2 = xH2*NHtot
    NCO = xCO*NHtot
    NHI = (1.0 - xHp - 2.0*xH2)*NHtot
    NTOT = NHtot*(1.0 + xHp - xH2 + xHe)

    return NTOT, NHI, NCO, NH2, NHp

def rotate(x,y,theta):
    xprime = x*cos(theta) - y*sin(theta)
    yprime = x*sin(theta) + y*cos(theta)
    return xprime, yprime

############################
# XYZ are cartesian coordinates centered at the Sun position. X axis points to the GC, Z axis up to north Galactic pole.
# xyz are cartesian coordinates centered at the GC with same orientations as XYZ.
# lbr are spherical coordinates centered at the Sun position (i.e., the usual Galactic coordinates).
# the following functions convert back and forth between the above coordinates
#############################

def XYZ2lbr(X,Y,Z):
    r = sqrt(X**2+Y**2+Z**2)
    l = arctan2(Y,X)
    b = pi/2 - arccos(Z/r)
    return l,b,r

def lbr2XYZ(l,b,r):
    X = r*sin(b+pi/2)*cos(l)
    Y = r*sin(b+pi/2)*sin(l)
    Z = r*cos(b+pi/2)
    return X,Y,Z

def xyz2XYZ(x,y,z,R0):
    return x+R0,y,z

def XYZ2xyz(X,Y,Z,R0):
    return X-R0,Y,Z

def xyz2lbr(x,y,z,R0):
    X,Y,Z = xyz2XYZ(x,y,z,R0)
    l,b,r = XYZ2lbr(X,Y,Z)
    return l,b,r

def lbr2xyz(l,b,r):
    X,Y,Z = lbr2XYZ(l,b,r)
    x,y,z = XYZ2xyz(X,Y,Z,R0)
    return x,y,z

def vxyz2vlbr(x,y,z,vx,vy,vz,R0):
    # see wiki "vector fields in spherical coords" for formulas
    X,Y,Z = xyz2XYZ(x,y,z,R0)
    l,b,r = XYZ2lbr(X,Y,Z)
    rhat = [sin(b+pi/2)*cos(l),sin(b+pi/2)*sin(l),cos(b+pi/2)]
    bhat = [cos(b+pi/2)*cos(l),cos(b+pi/2)*sin(l),-sin(b+pi/2)] 
    lhat = [-sin(l),cos(l),0] 
    vsun = 220.0
    vr = vx*rhat[0] + (vy-vsun)*rhat[1] + vz*rhat[2]
    vb = vx*bhat[0] + (vy-vsun)*bhat[1] + vz*bhat[2]
    vl = vx*lhat[0] + (vy-vsun)*lhat[1]
    return l,b,r,vl,vb,vr
  
def Rotate_around_u(x,y,z,u,alpha):
    # rotates x,y,z by angle alpha around axis passing through the origin defined by unit vector u =[ux,uy,uz]
    # found formulas googling...hope it works
    R_matrix = array([[cos(alpha) + u[0]**2 * (1-cos(alpha)),
                       u[0] * u[1] * (1-cos(alpha)) - u[2] * sin(alpha),
                       u[0] * u[2] * (1 - cos(alpha)) + u[1] * sin(alpha)],
                      [u[0] * u[1] * (1-cos(alpha)) + u[2] * sin(alpha),
                       cos(alpha) + u[1]**2 * (1-cos(alpha)),
                       u[1] * u[2] * (1 - cos(alpha)) - u[0] * sin(alpha)],
                      [u[0] * u[2] * (1-cos(alpha)) - u[1] * sin(alpha),
                       u[1] * u[2] * (1-cos(alpha)) + u[0] * sin(alpha),
                       cos(alpha) + u[2]**2 * (1-cos(alpha))]])
    xr,yr,zr = R_matrix.dot([x,y,z])
    return xr,yr,zr

def label_line(xxx, yyy, ax, line, label, color='0.5', fs=14, halign='left'):
    xdata, ydata = line.get_data()
    x1 = xdata[0]
    x2 = xdata[-1]
    y1 = ydata[0]
    y2 = ydata[-1]
    if halign.startswith('l'):
        xx = x1
        halign = 'left'
    elif halign.startswith('r'):
        xx = x2
        halign = 'right'
    elif halign.startswith('c'):
        xx = 0.5*(x1 + x2)
        halign = 'center'
    else:
        raise ValueError("Unrecognized `halign` = '{}'.".format(halign))
    yy = np.interp(xx, xdata, ydata)
    ylim = ax.get_ylim()
    # xytext = (10, 10)
    xytext = (0, 0)
    text = ax.annotate(label, xy=(xxx, yyy), xytext=xytext, textcoords='offset points',
    size=fs, color=color, zorder=1,
    horizontalalignment=halign, verticalalignment='center_baseline')
    sp1 = ax.transData.transform_point((x1, y1))
    sp2 = ax.transData.transform_point((x2, y2))
    rise = (sp2[1] - sp1[1])
    run = (sp2[0] - sp1[0])
    slope_degrees = np.degrees(np.arctan2(rise, run))
    text.set_rotation_mode('anchor')
    text.set_rotation(slope_degrees)
    ax.set_ylim(ylim)
    return text

#######################
# define class to manage arepo snapshot
#######################
class Snapshot:
    # use "chem=False" to load a snapshot without chemistry
    def __init__(self,isnap,path = '../',halfbox=120,halfboy=120,halfboz=120,R0=85.0,arepoLength=3.0856e20,arepoMass=1.911e33,arepoVel=1.0e5,chem=True):
        #
        self.isnap = isnap
        self.halfbox = halfbox
        self.halfboy = halfboy
        self.halfboz = halfboz
        self.R0 = R0
        self.chem = chem
        # convert from code units to cgs
        self.arepoLength = arepoLength
        self.arepoMass = arepoMass
        self.arepoVel = arepoVel
        self.arepoTime = self.arepoLength/self.arepoVel
        self.arepoDensity = self.arepoMass/self.arepoLength/self.arepoLength/self.arepoLength
        self.arepoEnergy= self.arepoMass*self.arepoVel*self.arepoVel
        self.arepoColumnDensity = self.arepoMass/self.arepoLength/self.arepoLength
        # read header
        self.path = path
        self.fname = self.path + 'whole_disk_%03d'%self.isnap
        # proj1 and 2 flag
        self.proj1 = False
        self.proj2 = False

    #####################
    # read full snapshot
    #####################
    def read_full(self):
        # read the full snapshot data
        self.data_gas, self.dummy, self.header = read_snapshot_file(self.fname)
        self.t = self.header['time'][0]
        self.x,self.y,self.z = self.data_gas['pos'].T
        self.x,self.y,self.z = self.x-self.halfbox, self.y-self.halfboy, self.z-self.halfboz
        self.vx,self.vy,self.vz = self.data_gas['vel'].T
        self.rho = self.data_gas['rho']
        self.masses = self.data_gas['mass']
        self.energy_per_unit_mass = self.data_gas['u_therm']
        self.volumes = self.data_gas['mass'] / self.data_gas['rho'] 
        #
        self.R = sqrt(self.x**2+self.y**2)
        self.tMyr = self.t*self.arepoTime/(60*60*24*365*1e6)
        self.rho_cgs = self.rho*self.arepoDensity
        self.energy_per_unit_mass_cgs = self.energy_per_unit_mass*(self.arepoEnergy/self.arepoMass)
        # number of particles
        self.n_gas = self.header['num_particles'][0]
        self.n_sink = self.header['num_particles'][5]
        self.rcell = 2*(self.volumes * 3 / (4 * pi))**(1.0/3.0)
        # chemistry stuff
        # nHtot = nHI + nHp + 2*nH2
        # nTOT = nHI + nH2 + nHp + ne + nHe
        if(self.chem):
            print('calculating chemistry stuff...')
            # chemical quantities are in cgs
            self.kpc_to_cm = 3.0856e21
            self.pc_to_cm  = 3.0856e18
            self.xHe = 0.1
            self.mp = 1.6726231e-24 # proton mass in g
            self.kb = 1.3806485e-16 # Boltzmann constant in g*cm^2/s^2/K 
            self.xH2, self.xHp, self.xCO = self.data_gas['chem'].T
            self.xHI = 1 - self.xHp -2*self.xH2
            self.nHtot = self.rho_cgs/((1. + 4.0 * self.xHe) * self.mp)
            self.nHp = self.xHp*self.nHtot
            self.nH2 = self.xH2*self.nHtot
            self.nCO = self.xCO*self.nHtot
            self.nHI = (1.0 - self.xHp - 2.0*self.xH2)*self.nHtot
            self.nHe = self.xHe*self.nHtot
            self.nTOT = self.nHtot*(1.0 + self.xHp - self.xH2 + self.xHe)
            self.mu = self.rho_cgs/(self.nTOT*self.mp)
            self.T = (2.0/3.0)*self.energy_per_unit_mass_cgs*self.mu*self.mp/self.kb
            self.cs_cgs = sqrt((5.0/3.0)*self.kb*self.T/self.mu/self.mp)
            self.cs_kms = self.cs_cgs/1e5
            self.Jeans_length_cm = Calc_LambdaJeans(self.rho_cgs,self.cs_kms)
            self.Jeans_length_100pc = self.Jeans_length_cm/self.kpc_to_cm*10
            self.Jeans_mass = Calc_MassJeans(self.rho_cgs,self.cs_kms)
            self.Jeans_number = self.Jeans_length_100pc/self.rcell
            self.Jeans_number_mass = self.Jeans_mass/self.masses
            # masses in [M]=M_sol
            self.massesH2 = (self.nH2*(2*self.mp)/self.arepoDensity)*self.volumes
            self.massesHI = (self.nHI*self.mp/self.arepoDensity)*self.volumes
            self.massesHp = (self.nHp*self.mp/self.arepoDensity)*self.volumes
            self.massesCO = (self.nCO*28*self.mp/self.arepoDensity)*self.volumes
            self.massesHe = (self.nHe*4*self.mp/self.arepoDensity)*self.volumes

    #####################
    # read sinks
    #####################
    def read_sinks(self):
        print('reading sinks...')
        self.fname_sink = self.path+'sink_snap_%03d'%isnap
        self.data_sink, self.t_sink, self.NSinksAllTasks = rsnap.read_sink_snap(self.fname_sink, MAXSNE=500, MAXACCRETIONEVENTS=50, longid=True)
        self.x_sink,self.y_sink,self.z_sink = self.data_sink['pos'].T
        self.x_sink,self.y_sink,self.z_sink = self.x_sink-self.halfbox, self.y_sink-self.halfboy, self.z_sink-self.halfboz
        self.vx_sink,self.vy_sink,self.vz_sink = self.data_sink['vel'].T
        self.masses_sink = self.data_sink['mass']
        self.R_sink = sqrt(self.x_sink**2+self.y_sink**2)
        self.age_sink = self.t_sink - self.data_sink['formationTime']
        self.age_sink_Myr = self.age_sink[:,0]*(self.arepoTime/(60*60*24*365*1e6))

    #####################
    # read projections
    #####################
    def read_proj1(self,foldername='proj1'):
        print('reading projection 1...')
        self.rho_proj1 = read_arepo_image(self.path+foldername+'density_proj_%03d'%self.isnap)
        self.xHp_proj1 = read_arepo_image(self.path+foldername+'xHP_proj_%03d'%self.isnap)
        self.xH2_proj1 = read_arepo_image(self.path+foldername+'xH2_proj_%03d'%self.isnap)
        self.xCO_proj1 = read_arepo_image(self.path+foldername+'xCO_proj_%03d'%self.isnap)
        self.xHI_proj1 = 1 - self.xHp_proj1 -2*self.xH2_proj1
        self.nHtot_proj1 = (self.rho_proj1 * self.arepoColumnDensity) / ((1. + 4.0 * self.xHe) * self.mp)
        self.nHp_proj1 = self.xHp_proj1*self.nHtot_proj1
        self.nH2_proj1 = self.xH2_proj1*self.nHtot_proj1
        self.nCO_proj1 = self.xCO_proj1*self.nHtot_proj1
        self.nHI_proj1 = (1.0 - self.xHp_proj1 - 2.0*self.xH2_proj1)*self.nHtot_proj1
        self.nTOT_proj1 = self.nHtot_proj1*(1.0 + self.xHp_proj1 - self.xH2_proj1 + self.xHe)
        self.proj1 = True

    def read_proj2(self,foldername='proj2'):
        print('reading projection 2...')
        self.rho_proj2 = read_arepo_image(self.path+foldername+'density_proj_%03d'%self.isnap)
        self.xHp_proj2 = read_arepo_image(self.path+foldername+'xHP_proj_%03d'%self.isnap)
        self.xH2_proj2 = read_arepo_image(self.path+foldername+'xH2_proj_%03d'%self.isnap)
        self.xCO_proj2 = read_arepo_image(self.path+foldername+'xCO_proj_%03d'%self.isnap)
        self.xHI_proj2 = 1 - self.xHp_proj2 -2*self.xH2_proj2
        self.nHtot_proj2 = (self.rho_proj2 * self.arepoColumnDensity) / ((1. + 4.0 * self.xHe) * self.mp)
        self.nHp_proj2 = self.xHp_proj2*self.nHtot_proj2
        self.nH2_proj2 = self.xH2_proj2*self.nHtot_proj2
        self.nCO_proj2 = self.xCO_proj2*self.nHtot_proj2
        self.nHI_proj2 = (1.0 - self.xHp_proj2 - 2.0*self.xH2_proj2)*self.nHtot_proj2
        self.nTOT_proj2 = self.nHtot_proj2*(1.0 + self.xHp_proj2 - self.xH2_proj2 + self.xHe)
        self.proj2 = True

    #####################
    # calc lbv points
    #####################
    def calc_lbv(self):
        print('calculating lbv points...')
        self.l,self.b,self.r,self.vl,self.vb,self.vr = vxyz2vlbr(self.x,self.y,self.z,self.vx,self.vy,self.vz,self.R0)
        if(self.n_sink):
            self.l_sink,self.b_sink,self.r_sink,self.vl_sink,self.vb_sink,self.vr_sink = vxyz2vlbr(self.x_sink,self.y_sink,self.z_sink,self.vx_sink,self.vy_sink,self.vz_sink,self.R0)

    #####################
    # rotate the snapshot by a given angle
    #####################
    def rotate_full(self,theta):
        self.x,self.y = rotate(self.x,self.y,theta)
        self.vx,self.vy = rotate(self.vx,self.vy,theta)
        if(self.n_sink):
            self.x_sink,self.y_sink = rotate(self.x_sink,self.y_sink,theta)
            self.vx_sink,self.vy_sink = rotate(self.vx_sink,self.vy_sink,theta)

    #####################
    # rotate projections by given angle
    #####################
    def rotate_proj(self,theta):
        if(self.proj1):
            self.rho_proj1 = ndimage.rotate(self.rho_proj1, np.degrees(theta), reshape=False,order=1)
            self.xHp_proj1 = ndimage.rotate(self.xHp_proj1, np.degrees(theta), reshape=False,order=1)
            self.xH2_proj1 = ndimage.rotate(self.xH2_proj1, np.degrees(theta), reshape=False,order=1)
            self.xCO_proj1 = ndimage.rotate(self.xCO_proj1, np.degrees(theta), reshape=False,order=1)
            self.xHI_proj1 = ndimage.rotate(self.xHI_proj1, np.degrees(theta), reshape=False,order=1)
            self.nHtot_proj1 = ndimage.rotate(self.nHtot_proj1, np.degrees(theta), reshape=False,order=1)
            self.nHp_proj1 = ndimage.rotate(self.nHp_proj1, np.degrees(theta), reshape=False,order=1)
            self.nH2_proj1 = ndimage.rotate(self.nH2_proj1, np.degrees(theta), reshape=False,order=1)
            self.nCO_proj1 = ndimage.rotate(self.nCO_proj1, np.degrees(theta), reshape=False,order=1)
            self.nHI_proj1 = ndimage.rotate(self.nHI_proj1, np.degrees(theta), reshape=False,order=1)
            self.nTOT_proj1 = ndimage.rotate(self.nTOT_proj1, np.degrees(theta), reshape=False,order=1)
        #
        if(self.proj2):
            self.rho_proj2 = ndimage.rotate(self.rho_proj2, np.degrees(theta), reshape=False,order=1)
            self.xHp_proj2 = ndimage.rotate(self.xHp_proj2, np.degrees(theta), reshape=False,order=1)
            self.xH2_proj2 = ndimage.rotate(self.xH2_proj2, np.degrees(theta), reshape=False,order=1)
            self.xCO_proj2 = ndimage.rotate(self.xCO_proj2, np.degrees(theta), reshape=False,order=1)
            self.xHI_proj2 = ndimage.rotate(self.xHI_proj2, np.degrees(theta), reshape=False,order=1)
            self.nHtot_proj2 = ndimage.rotate(self.nHtot_proj2, np.degrees(theta), reshape=False,order=1)
            self.nHp_proj2 = ndimage.rotate(self.nHp_proj2, np.degrees(theta), reshape=False,order=1)
            self.nH2_proj2 = ndimage.rotate(self.nH2_proj2, np.degrees(theta), reshape=False,order=1)
            self.nCO_proj2 = ndimage.rotate(self.nCO_proj2, np.degrees(theta), reshape=False,order=1)
            self.nHI_proj2 = ndimage.rotate(self.nHI_proj2, np.degrees(theta), reshape=False,order=1)
            self.nTOT_proj2 = ndimage.rotate(self.nTOT_proj2, np.degrees(theta), reshape=False,order=1)

    #####################
    # create interpolating functions
    #####################
    def create_interpolating_functions(self):
        print('creating interpolating functions...')
        points = np.vstack((self.x,self.y,self.z)).T
        self.frho = NearestNDInterpolator(points,self.rho)
        self.fT = NearestNDInterpolator(points,self.T)
        self.nH2 = NearestNDInterpolator(points,self.nH2)

    #####################
    # reduce size of arrays
    #####################
    def reduce_size(self,DD=10):
        print('reducing sizes...')
        self.x,self.y,self.z = self.x[::DD], self.y[::DD], self.z[::DD]
        self.vx,self.vy,self.vz = self.vx[::DD], self.vy[::DD], self.vz[::DD]
        self.rho = self.rho[::DD]
        self.masses = self.masses[::DD]
        self.xH2, self.xHp, self.xCO = self.xH2[::DD], self.xHp[::DD], self.xCO[::DD] 
        self.energy_per_unit_mass = self.energy_per_unit_mass[::DD]
        self.volumes = self.volumes[::DD]
        self.rho_cgs = self.rho_cgs[::DD]
        self.T = self.T[::DD]
        self.nHtot = self.rho_cgs/((1. + 4.0 * self.xHe) * self.mp)
        self.nHp = self.nHp[::DD]
        self.nH2 = self.nH2[::DD]
        self.nCO = self.nCO[::DD]
        self.nHI = self.nHI[::DD]
        self.nHe = self.nHe[::DD]
        self.nTOT = self.nTOT[::DD]
    
    #####################
    # cut everything that does not satisfy a condition (e.g. condition = sqrt(x**2+y**2)<R**2)
    #####################
    def impose_condition(self,condition):
        CC = condition
        self.x,self.y,self.z = self.x[CC], self.y[CC], self.z[CC]
        self.vx,self.vy,self.vz = self.vx[CC], self.vy[CC], self.vz[CC]
        self.rho = self.rho[CC]
        self.masses = self.masses[CC]
        self.xH2, self.xHp, self.xCO = self.xH2[CC], self.xHp[CC], self.xCO[CC] 
        self.energy_per_unit_mass = self.energy_per_unit_mass[CC]
        self.volumes = self.volumes[CC]
        self.rho_cgs = self.rho_cgs[CC]
        self.massesH2 = self.massesH2[CC]
        self.massesHI = self.massesHI[CC]
        self.massesHp = self.massesHp[CC]
        self.massesCO = self.massesCO[CC]
        self.massesHe = self.massesHe[CC]