from abc import ABC
import faiss


class NearestNeighborSearcher(ABC):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def build_index(self, points):
        raise NotImplementedError
    
    def query(self, query_points, k, return_distances):
        raise NotImplementedError
    

class ExactFaissNearestNeighborSearcher(NearestNeighborSearcher):
    def __init__(self, device):
        super().__init__(device)
        self.index = None

        if self.device == 'cuda': 
            print(f'Using FAISS with {faiss.get_num_gpus()} GPU(s).')
        else: print('Using FAISS on CPU.')
        
    def build_index(self, points):
        index = faiss.IndexFlatL2(points.shape[1])
       
        if self.device == 'cuda':
            index = faiss.index_cpu_to_all_gpus(index)
    
        index.add(points.numpy())
        self.index = index  

    def query(self, query_points, k, return_distances = False):
        D, I = self.index.search(query_points.numpy(), k)
        if return_distances:
            return I, D
        return I
    
class ApproximateFaissNearestNeighborSearcher(NearestNeighborSearcher):
    def __init__(self, device, nlist, nprobe):
        super().__init__(device)
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None

        if self.device == 'cuda': 
            print(f'Using FAISS with {faiss.get_num_gpus()} GPU(s).')
        else: print('Using FAISS on CPU.')
        
    def build_index(self, points):
        d = points.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, self.nlist)
    
        if self.device == 'cuda':
            index = faiss.index_cpu_to_all_gpus(index)
    
        index.train(points.numpy())
        index.add(points.numpy())

        index.nprobe = self.nprobe

        self.index = index

    def query(self, query_points, k, return_distances = False):
        D, I = self.index.search(query_points.numpy(), k)
        if return_distances:
            return I, D 
        return I