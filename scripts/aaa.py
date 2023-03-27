from tqdm import tqdm

N = 100000
r = range(N)

loop=tqdm(r,leave=False)
for x in loop:
    loop.set_description(f'x={x}/{N}')

