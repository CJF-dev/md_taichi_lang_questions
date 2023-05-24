import taichi as ti
import numpy as np

'''
How to Accelerate this condition checking with vector operator
'''
cell_size = [62.0, 62.0, 62.0]
cutoff = 9.0
cutoffa = 12.73
cutoffb = 15.59
num_cells = 216
num_atoms = 2048
xnc = 6
ync = 6
znc = 6
nc = 1
 

@ti.kernel
def build_list_kernel(rlpos: ti.template(), nb_list: ti.template()):
    for x in rlpos:
        home_cell_id = x % 100
        rij = rlpos[x]

        # TODO:
        # Accelerate this condition checking with vector operator
        if rij[0] > cell_size[0]/2. :
            rij[0] -= cell_size[0]
        elif rij[0] < -cell_size[0]/2. :
            rij[0] += cell_size[0]
        
        if rij[1] > cell_size[1]/2. :
            rij[1] -= cell_size[1]
        elif rij[1] < -cell_size[1]/2. :
            rij[1] += cell_size[1]

        if rij[2] > cell_size[2]/2. :
            rij[2] -= cell_size[2]
        elif rij[2] < -cell_size[2]/2. :
            rij[2] += cell_size[2]

        # TODO:
        # Accelerate this condition checking with vector operator
        if abs(rij[0]) < cutoff and abs(rij[1]) < cutoff and abs(rij[2]) < cutoff:
            if abs(rij[0])+abs(rij[1]) < cutoffa \
                    and abs(rij[0])+abs(rij[2]) < cutoffa and abs(rij[1])+abs(rij[2]) < cutoffa:
                if abs(rij[0]) + abs(rij[1]) + abs(rij[2]) < cutoffb :
                    nb_list[home_cell_id].append(x)

def build_list():
    nb_list = ti.field(ti.i32)
    block = ti.root.dense(ti.i, num_cells)
    values = block.dynamic(ti.j, num_atoms, chunk_size=20)
    values.place(nb_list)

    rlpos_np = np.random.randn(num_atoms, 3) * 10
    rlpos = ti.Vector.field(n=3, dtype=ti.f64, shape=num_atoms)
    rlpos.from_numpy(rlpos_np)
    build_list_kernel(rlpos, nb_list)

if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    build_list()