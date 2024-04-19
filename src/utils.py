from time import perf_counter


def divide_chunks(l, n):
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
        

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        tic = perf_counter()
        result = func(*args, **kwargs)
        toc = perf_counter()
        print(f"Function {func.__qualname__} took {toc - tic:.4f} seconds to execute")
        return result
    return wrapper
