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