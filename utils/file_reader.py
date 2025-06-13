import numpy as np
import os
import re
from dscribe.descriptors import SOAP

def castep_cell_reader(filename):
    fh = open(filename, 'r')
    line = '-'
    while line != '':
        line = fh.readline().lower()
        if re.search('^%block lattice_cart', line):
            cell = list();
            line = fh.readline().lower()
            if 'ang' in line:
                for _ in range(3):
                    temp = fh.readline().strip().split()
                    cell.append(list(map(float, temp)))
            elif 'bohr' in line:
                unit_conversion = 1.0
                for _ in range(3):
                    temp = fh.readline().strip().split()
                    cell.append(list(map(lambda x: unit_conversion*float(x), temp)))
            else:
                cell.append(list(map(float, line.strip().split())))
                for _ in range(2):
                    temp = fh.readline().strip().split()
                    cell.append(list(map(float, temp)))
        
        if re.search('^%block positions_abs', line):
            species = []; positions = [];
            line = fh.readline()
            while not re.search('^%endblock positions_abs', line.lower()):
                species.append(line.strip().split()[0])
                positions.append( list( map(float, line.strip().split()[1:4] ) ) )
                line = fh.readline()
    return cell, species, positions

def density_reader(filename, path_to_dir = None):
    fh = open(path_to_dir + '/' + filename, 'r')
    line = '-'
    while line != '':
        line = fh.readline()
        if 'iteration' in line:
            step = int(line.strip().split()[-1])
        if '3D fine grid' in line:
            dnx, dny, dnz = list(map(int, line.strip().split()[-3:]))
            density = np.zeros([dnx, dny, dnz], dtype = np.float32)
        if 'Data format' in line:
            for _ in range(dnx*dny*dnz):
                line = fh.readline()
                cline = line.strip().split()
                ix, iy, iz = list(map(int, cline[:3]))
                density[ix-1,iy-1,iz-1] = np.float32(cline[3])
    return density, step

def potential_reader(file):
    cell = []; abc = []; angles = []; 
    dng = [];  indx = []; pot = [];
    fh = open(file,'r')
    flag = True
    while flag:
        line = fh.readline()
        if 'Real Lattice' in line:
            for j in range(3):
                line = fh.readline()
                cell += [list(map(float, line.strip().split()[:3]))]
                abc  += [float(line.strip().split()[5])]
                angles += [float(line.strip().split()[8])]
                
        if 'fine FFT grid along' in line:
            dng += list(map(int, line.strip().split()[:3]))
        if 'END header' in line:
            #tot_dng = np.prod(dng).astype(int)
            tot_dng = np.prod(dng)
            line = fh.readline()
            for j in range(tot_dng):
                line = fh.readline()
                indx += [list(map(lambda x: int(x) - 1, line.strip().split()[:3]))]
                pot += [float(line.strip().split()[3])]
            flag = False
    potential = np.zeros(dng, dtype = np.float64)
    for i in range(len(pot)):
        potential[*indx[i]] = pot[i]

    return potential

def data_loader(path_to_dirs):
    density_out = list(); density_in = list();
    iterations = list(); dirs = list(); potentials = list()
    potential_exit = False

    for dirc in os.listdir(path_to_dirs):
        if dirc != 'run.sh' and dirc != '0':
            path_to_dir = os.path.join(path_to_dirs, dirc)
            files = os.listdir(path_to_dir)
            dirs.append(dirc)
            den_in = []; den_out = []; iters = []
            for file in files:
                if 'density_in' in file:
                    dnt, itr = density_reader(file, path_to_dir)
                    den_in.append(dnt)
                elif 'density_out' in file:
                    dnt, itr = density_reader(file, path_to_dir)
                    den_out.append(dnt)
                    iters.append(itr)
                elif 'pot_fmt' in file:
                    potential_exit = True
                    potential = potential_reader(path_to_dir + '/' + file)
                    potentials += [potential]
            density_in.append(den_in); density_out.append(den_out);
            iterations.append(iters); dirs.append(dirc)
    if potential_exit:
        print(f"Potential files found in the directory.")
        return density_in, density_out, potentials, iterations
    else:
        return density_in, density_out, iterations

def data_transform(scaler, density_in, density_out, potential, iterations):
    X_temp = list(); Y_temp = list(); Z_temp = list();
    for i in range(len(iterations)):
        indx_srt = np.argsort( iterations[i] )
        X_temp += [density_in[i][indx_srt[0]].ravel(order = 'C')]
        Y_temp += [density_out[i][indx_srt[-1]].ravel(order = 'C')]
        ###
        Z_temp += [density_in[i][indx_srt[1]].ravel(order = 'C')]
    scaler.fit( np.array(X_temp) )
    X_temp = scaler.transform(X_temp)
    Y_temp = scaler.transform(Y_temp)
    ###
    Z_temp = scaler.transform(Z_temp)
    indx = np.unravel_index(range(X_temp.shape[1]), density_out[0][0].shape, order = 'C')
    X = np.zeros((X_temp.shape[0],2,*density_out[0][0].shape), dtype = np.float64)
    Y = np.zeros((Y_temp.shape[0],*density_out[0][0].shape), dtype = np.float64)
    if potential is not None:
        for i,frame in enumerate(X_temp):
            X[i,0] = potential[i]
            X[i,1,indx[0],indx[1],indx[2]] = frame
            Y[i,  indx[0],indx[1],indx[2]] = Y_temp[i]
    else:
        for i,frame in enumerate(X_temp):
            X[i,0,indx[0],indx[1],indx[2]] = Z_temp[i]
            X[i,1,indx[0],indx[1],indx[2]] = frame
            Y[i,  indx[0],indx[1],indx[2]] = Y_temp[i]
    return X, Y

def inverse_transform(scaler, pred):
    pred_inv = scaler.inverse_transform( pred.ravel(order = 'C').reshape(1,-1) )
    indx = np.unravel_index(range(np.prod(pred.shape)), pred.shape[:], order = 'C')
    scaled = np.zeros(pred.shape[:], dtype = np.float64)
    scaled[indx[0], indx[1], indx[2]] = pred_inv
    return scaled

def feature_vectors(structure, cell, dnx, dny, dnz, weighting=None):
    dn_vec = np.array(cell)/np.array([dnx, dny, dnz])
    grid = list(); grid_indx = list()
    for ix in range(dnx):
        for iy in range(dny):
            for iz in range(dnz):
                grid.append(dn_vec@[ix, iy, iz])
                grid_indx.append([ix,iy,iz])
    
    distance = structure.get_distance(0,1)
    rcut = 2*distance; lmax = 8; nmax = 8
    if weighting is None:
        feature = SOAP(species=structure.numbers, r_cut=rcut, l_max=lmax, n_max=nmax,
                       sparse=False, periodic=True)
    else:
        feature = SOAP(species=structure.numbers, r_cut=rcut, l_max=lmax, n_max=nmax,
                       sparse=False, periodic=True,
                       weighting={"function":"poly","r0":2,"c":2,"m":2})
        
    feature_vectors = feature.create(system=structure, centers=grid)
    return feature_vectors, grid_indx