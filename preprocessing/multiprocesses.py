from multiprocessing import Process, Queue
import time
import sys

def process_params(params):
    """Process the params"""
    # print("Starting params: %s" % params['id'])

    time.sleep(2)
    print("Finishing params: %s" % params)


def reader_proc(queue):
    """Read from the queue; this spawns as a separate Process"""
    while True:
        msg = queue.get()  # Read from the queue and do nothing
        if msg == "DONE":
            break
        process_params(msg)

def writer(params, num_of_reader_procs, queue):
    """Write integers into the queue.  A reader_proc() will read them from the queue"""
    for param in params:
        queue.put(param)  # Put 'count' numbers into queue

    ### Tell all readers to stop...
    for ii in range(0, num_of_reader_procs):
        queue.put("DONE")

def start_reader_procs(qq, num_of_reader_procs):
    """Start the reader processes and return all in a list to the caller"""
    all_reader_procs = list()
    for ii in range(0, num_of_reader_procs):
        reader_p = Process(target=reader_proc, args=((qq),))
        reader_p.daemon = True
        reader_p.start()  # Launch reader_p() as another proc
        all_reader_procs.append(reader_p)
    return all_reader_procs

def generate_params():
    """Generate parameters to pass to process_params()"""
    param_list = list()
    for group in [10, 20, 30, 40, 50]:
        dict_of_params = dict()
        dict_of_params["preprocess"] = "PCA"
        dict_of_params["groupSize"] = group
        param_list.append(dict_of_params)
    return param_list

if __name__ == "__main__":
    num_of_reader_procs = 5
    qq = Queue()  # writer() writes to qq from _this_ process
    all_reader_procs = start_reader_procs(qq, num_of_reader_procs)
    writer(generate_params(), len(all_reader_procs), qq) 
    for idx, a_reader_proc in enumerate(all_reader_procs):
        a_reader_proc.join()
    print("Done")