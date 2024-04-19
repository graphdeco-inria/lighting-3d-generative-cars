import pickle
import torch
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA


@torch.no_grad()
def get_pca_components(mapping_network, pca_path):
    mapping_network.eval()
    num_samples = int(1e4)
    batch_size = 400
    num_components = batch_size
    inc_pca = IncrementalPCA(num_components)

    torch.manual_seed(0)
    zs = torch.randn((num_samples, 512), device="cuda")

    c = torch.zeros((batch_size, 25), device="cuda")
    count = 0
    with tqdm(initial=0, total=num_samples) as pbar:
        while count < num_samples:
            with torch.no_grad():
                w = mapping_network(zs[count : count + batch_size], c)
            inc_pca.partial_fit(w[:, 0, :].cpu().numpy())
            count += batch_size
            pbar.update(batch_size)

    # Save PCA
    with open(pca_path, "wb") as f:
        pickle.dump(inc_pca, f)
