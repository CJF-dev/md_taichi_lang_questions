# Taichi并行化加速MD相关问题汇总
## 1.多个没有数据依赖的kernel同时运行的问题
下面代码中，两个ti.kernel并无数据依赖关系，taichi中有无方法可以使两个或多个kernel并行？

```python
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
```

## 2. 如何取出sparse list，拼接成dense list

目前只能通过遍历sparse list中的每个元素后，重新append进新的list方法实现，有无更高效的办法，例如可否直接做list拼接实现？

```python
import taichi as ti
import time

'''
1. Faster way to convert sparse field to dense list.
2. How to reduce compile time of get_dense_kernel.
'''

num_cells = 216

@ti.kernel
def sparse_kernel(values: ti.template(), length: ti.template()):
    for i in range(values.shape[0]):
        for j in range(ti.random(ti.i32) % 10):
            values[i].append(i*j)

        length[i] = values[i].length()
        # print(f"len: {values[i].length()}")

@ti.kernel
def get_dense_kernel(sparse_list: ti.template(), dense_list: ti.template()):
    for i in range(sparse_list.shape[0]):
        for j in range(sparse_list[i].length()):
            dense_list[0].append(sparse_list[i, j])


def run_kernel(sparse_list, length):
    start_time = time.time()
    sparse_kernel(sparse_list, length)
    print(f"[Taichi] time for sparse: {(time.time() - start_time): 0.4f} sec")

    start_time = time.time()
    num_items = length.to_numpy().sum()
    dense_list = ti.field(ti.i32)
    block = ti.root.dense(ti.i, 1)
    values = block.dynamic(ti.j, num_items, chunk_size=4)
    values.place(dense_list)

    get_dense_kernel(sparse_list, dense_list)
    print(f"[Taichi] time for sparse: {(time.time() - start_time): 0.4f} sec")

    print(f"dense_list: {dense_list.to_numpy().shape}")


def main():
    sparse_list = ti.field(ti.i32)
    block = ti.root.dense(ti.i, num_cells)
    values = block.dynamic(ti.j, 2048, chunk_size=4)
    values.place(sparse_list)

    length = ti.field(ti.int32, shape=num_cells)

    run_kernel(sparse_list, length)
    run_kernel(sparse_list, length)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    main()
```

## 3. Vec 条件指令加速

以下代码中条件检查部分，能否通过向量运算加速？

```python
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
```
