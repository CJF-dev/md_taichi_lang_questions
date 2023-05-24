import taichi as ti

''' How launch the first_kernel and second_kernel at the same time?
'''

@ti.kernel
def first_kernel(rlpos: ti.template()):
    for i in rlpos:
        rlpos[i] = i + 1

@ti.kernel
def second_kernel(cell_list: ti.template()):
    for i in cell_list:
        cell_list[i] = i + 2

def main():
    size = 100
    rlpos = ti.field(dtype=ti.f32, shape=size)
    cell_list = ti.field(dtype=ti.f32, shape=size)

    first_kernel(rlpos)
    second_kernel(cell_list)

if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    main()