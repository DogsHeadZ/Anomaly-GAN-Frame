import tqdm
try:
    with tqdm(range(90)) as t:
        for i in t:
            print(i)
except KeyboardInterrupt:
    t.close()
    raise
t.close()