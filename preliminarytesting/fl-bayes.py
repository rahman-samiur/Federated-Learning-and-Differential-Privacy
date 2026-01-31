from tqdm import tqdm
import random
from torch.distributions import Binomial
from scipy.stats import t as t_distribution
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch
import warnings

# Ignore only UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8

        # Calculate flattened size
        self.flat_size = 64 * 8 * 8

        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # (B, 3, 32, 32)
        x = self.pool1(F.relu(self.conv1(x)))  # (B, 32, 16, 16)
        x = self.pool2(F.relu(self.conv2(x)))  # (B, 64, 8, 8)
        x = x.view(-1, self.flat_size)        # (B, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalize with mean and stddev of CIFAR-10
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

    def prepare_data(self):
        # Download the dataset
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        # Called on every GPU
        if stage == "fit" or stage is None:
            # Load full training set
            full_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            # Split into 45k train, 5k validation
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [45000, 5000]
            )

        if stage == "test" or stage is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )



class Client:
    def __init__(self, client_id, local_data_loader):
        self.client_id = client_id
        self.loader = local_data_loader
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self, global_model, local_epochs):
        """ Trains a local model and returns the update vector """
        # Create a local copy of the global model
        local_model = SimpleCNN().to(self.device)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Get the starting model vector
        start_vec = parameters_to_vector(
            local_model.parameters()).clone().detach()

        for epoch in range(local_epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                logits = local_model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        # Get the final model vector
        # Get the final model vector
        end_vec = parameters_to_vector(local_model.parameters()).clone().detach()

        # Return the *update* (delta)
        # This delta is the "gradient" for our accountant
        client_update_vec = end_vec - start_vec
        return client_update_vec


class BayesianAccountant_FL:
    def __init__(self, m_samples: int, lambda_val: int, estimator_gamma: float = 1e-15):
        # ... (Same __init__ as before) ...
        if not isinstance(lambda_val, int) or lambda_val <= 0:
            raise ValueError("lambda_val (λ) must be a positive integer.")
        if m_samples < 2:
            raise ValueError("m_samples must be at least 2.")

        self.m_samples = m_samples
        self.lambda_val = lambda_val
        self.estimator_gamma = estimator_gamma
        self.costs = []

        df = self.m_samples - 1
        self.t_dist_factor = t_distribution.ppf(
            1 - self.estimator_gamma, df=df
        ) / math.sqrt(df)

    def _compute_binomial_expectation(self, d_squared, q, sigma_val):
        # ... (Same _compute_binomial_expectation as before) ...
        # This computes Z_x = E_k[ exp( (k^2-k)/(2*σ^2) * ||Δ_i - Δ_j||^2 ) ]

        # --- Compute Z_L (left side) ---
        lambda_L = self.lambda_val + 1
        k_L = torch.arange(lambda_L + 1, device=d_squared.device)
        probs_L = Binomial(lambda_L, q).log_prob(k_L).exp()
        term_L_exp = ((k_L**2 - k_L) / (2 * sigma_val**2)) * d_squared
        Z_L = (probs_L * term_L_exp.clamp(max=20).exp()).sum()

        # --- Compute Z_R (right side) ---
        lambda_R = self.lambda_val
        k_R = torch.arange(lambda_R + 1, device=d_squared.device)
        probs_R = Binomial(lambda_R, q).log_prob(k_R).exp()
        term_R_exp = ((k_R**2 + k_R) / (2 * sigma_val**2)) * d_squared
        Z_R = (probs_R * term_R_exp.clamp(max=20).exp()).sum()

        return torch.max(Z_L, Z_R)

    def compute_step_cost(self,
                          client_update_pairs: list,
                          q_subsampling: float,
                          clip_C: float,
                          noise_sigma: float):
        """
        Computes the privacy cost for one FL round.
        
        Args:
            client_update_pairs (list): A list of (delta_i, delta_j) tuples.
            q_subsampling (float): The client sampling probability (K/N).
            clip_C (float): The clipping norm for client updates.
            noise_sigma (float): The noise multiplier.
        """
        sigma_val = noise_sigma * clip_C

        Z_samples = []
        for delta_i, delta_j in client_update_pairs:
            # --- 1. Clip the updates (as the mechanism would) ---
            norm_i = torch.norm(delta_i, p=2)
            norm_j = torch.norm(delta_j, p=2)

            clipped_i = delta_i / torch.clamp(norm_i / clip_C, min=1.0)
            clipped_j = delta_j / torch.clamp(norm_j / clip_C, min=1.0)

            # --- 2. Compute d^2 = ||Δ_i_clipped - Δ_j_clipped||^2 ---
            d_squared = torch.sum((clipped_i - clipped_j)**2)

            # --- 3. Compute Z_i for this pair ---
            Z_i = self._compute_binomial_expectation(
                d_squared, q_subsampling, sigma_val
            )
            Z_samples.append(Z_i)

        Z_samples = torch.stack(Z_samples)

        # --- 4. Apply the Estimator (Eq 14) ---
        M_t = Z_samples.mean()
        S_t = Z_samples.std()

        if torch.isnan(S_t) or S_t == 0:
            S_t = torch.tensor(0.0)  # Handle single sample case

        estimated_expectation = M_t + self.t_dist_factor * S_t
        c_t = torch.log(estimated_expectation)

        self.costs.append(c_t)

    def get_epsilon(self, delta: float):
        # ... (Same get_epsilon as before) ...
        if not self.costs:
            return 0.0
        beta = delta - self.estimator_gamma
        if beta <= 0:
            raise ValueError(f"Target delta ({delta}) must be > "
                             f"estimator_gamma ({self.estimator_gamma}).")
        total_cost = torch.stack(self.costs).sum()
        epsilon_mu = (total_cost - torch.log(torch.tensor(beta))
                      ) / self.lambda_val
        return epsilon_mu.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Simulation Setup ---
    NUM_CLIENTS = 10
    CLIENTS_PER_ROUND = 2     # K
    LOCAL_EPOCHS = 2
    GLOBAL_ROUNDS = 3

    # --- 2. BDP-FL Setup ---
    CLIP_C = 1.0               # Clipping norm for client deltas
    NOISE_SIGMA = 2          # Noise multiplier
    BDP_M_SAMPLES = 8          # 'm' pairs to sample for BDP per round
    BDP_LAMBDA = 5             # 'λ'
    BDP_GAMMA = 1e-6           # 'γ'
    TARGET_DELTA = 1e-5

    # --- 3. Data and Client Setup ---
    # Load full dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=CIFAR10DataModule().transform
    )

    # Create non-IID data split (e.g., 500 samples per client)
    data_per_client = 50000//NUM_CLIENTS
    split_lengths = [data_per_client] * NUM_CLIENTS

    # Add a check for safety
    if sum(split_lengths) != len(full_dataset):
        raise ValueError(f"Data split mismatch: Sum of splits {sum(split_lengths)} "
                         f"does not equal dataset length {len(full_dataset)}.")

    client_datasets = random_split(
        full_dataset, 
        split_lengths
    )
    
    clients = []
    for i in range(NUM_CLIENTS):
        loader = DataLoader(client_datasets[i], batch_size=32, shuffle=True)
        clients.append(Client(client_id=i, local_data_loader=loader))

    print(f"Created {len(clients)} clients.")

    # --- 4. Server Initialization ---
    global_model = SimpleCNN().to(device)

    # The BDP accountant
    accountant = BayesianAccountant_FL(
        m_samples=BDP_M_SAMPLES,
        lambda_val=BDP_LAMBDA,
        estimator_gamma=BDP_GAMMA
    )

    q_subsampling = CLIENTS_PER_ROUND / NUM_CLIENTS

    # --- 5. Main Training Loop ---
    for round in tqdm(range(GLOBAL_ROUNDS), desc="Global Rounds"):

        # --- BDP Accounting Step ---
        # "Sample m pairs of client updates to estimate cost"
        # In a sim, we just pick m pairs from our list
        update_pairs = []
        for _ in range(BDP_M_SAMPLES):
            c1, c2 = random.sample(clients, 2)
            # 1 epoch for speed
            delta_i = c1.train(global_model, local_epochs=1)
            delta_j = c2.train(global_model, local_epochs=1)
            update_pairs.append((delta_i.to(device), delta_j.to(device)))

        accountant.compute_step_cost(
            update_pairs, q_subsampling, CLIP_C, NOISE_SIGMA
        )

        # --- DP-FedAvg Aggregation Step ---
        # "Sample K clients for actual model update"
        selected_clients = random.sample(clients, CLIENTS_PER_ROUND)

        round_updates = []
        for client in selected_clients:
            delta_vec = client.train(global_model, LOCAL_EPOCHS).to(device)

            # --- Apply DP-SGD Mechanism ---
            # 1. Clip
            norm = torch.norm(delta_vec, p=2)
            scale = torch.clamp(norm / CLIP_C, min=1.0)
            clipped_delta = delta_vec / scale

            round_updates.append(clipped_delta)

        # 2. Average (FedAvg)
        if round_updates:
            avg_delta = torch.stack(round_updates).mean(dim=0)

            # 3. Add Noise
            # Noise is N(0, (σC)^2 / K^2) on the *client* update,
            # or N(0, (σC)^2 / K) on the *averaged* update.
            # We add it to the average.
            noise_std = (NOISE_SIGMA * CLIP_C) / CLIENTS_PER_ROUND
            noise = torch.normal(
                0, noise_std, size=avg_delta.shape, device=device
            )
            final_delta = avg_delta + noise

            # --- Update Global Model ---
            with torch.no_grad():
                current_vec = parameters_to_vector(global_model.parameters())
                vector_to_parameters(current_vec + final_delta, global_model.parameters())

        # Log BDP epsilon
        if round % 5 == 0:
            eps_mu = accountant.get_epsilon(TARGET_DELTA)
            print(f"\nRound {round}: BDP Epsilon (ε_μ) = {eps_mu:.3f}")

    # --- End of Training ---
    final_epsilon_mu = accountant.get_epsilon(TARGET_DELTA)
    print(f"\n--- Training Complete ---")
    print(f"Final BDP Guarantee: (ε_μ = {final_epsilon_mu:.4f}, δ = {TARGET_DELTA})")
