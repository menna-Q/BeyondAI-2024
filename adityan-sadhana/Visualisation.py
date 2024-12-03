import numpy as np
from sgp4.api import Satrec, jday
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def propagate_tle(tle_data, timesteps=40):
    frames = []
    for tle in tle_data:
        sat = Satrec.twoline2rv(tle[0], tle[1])
        frame = []
        jd, fr = jday(2000, 1, 1, 0, 0, 0)

        for t in range(timesteps):
            jd_timestep = jd + t * (1 / 1440)
            _, position, _ = sat.sgp4(jd_timestep, fr)
            if position:
                frame.append(position)
            else:
                frame.append([None, None, None])
        frames.append(frame)

    frames = np.array(frames)
    frames = np.nan_to_num(frames, nan=0.0)
    return frames

from sklearn.preprocessing import StandardScaler

def normalize_frames(frames):
    scaler = StandardScaler()
    frames_reshaped = frames.reshape(-1, frames.shape[-1])
    frames_normalized = scaler.fit_transform(frames_reshaped)
    return frames_normalized.reshape(frames.shape), scaler

def prepare_graph_data(frames):
    graphs = []
    num_timestamps = frames.shape[1]

    for t in range(num_timestamps):
        positions_at_timestep = frames[:, t]
        num_nodes = len(positions_at_timestep)

        edge_index = torch.tensor(
            [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j],
            dtype=torch.long
        ).t().contiguous()

        x = torch.tensor(positions_at_timestep, dtype=torch.float)
        x = torch.nan_to_num(x, nan=0.0)
        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs

class DGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.nan_to_num(x, nan=0.0)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return torch.nan_to_num(x, nan=0.0)

def train_model(graphs, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGNN(input_dim=3, hidden_dim=16, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.HuberLoss(delta=1.0)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for graph in graphs:
            graph = graph.to(device)
            optimizer.zero_grad()
            output = model(graph)
            loss = criterion(output, graph.x)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
    return model

def generate_predictions(model, graphs):
    model.eval()
    predictions = []
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to('cuda' if torch.cuda.is_available() else 'cpu')
            output = model(graph).cpu().numpy()
            predictions.append(output)
    predictions = np.array(predictions)
    return predictions

tle_data, satellite_names = load_tle("/kaggle/input/datafortraining/Data.txt")
frames = propagate_tle(tle_data, timesteps=40)
frames, scaler = normalize_frames(frames)
graphs = prepare_graph_data(frames)

model = train_model(graphs, epochs=100)

tle_test_data, satellite_names = load_tle("/kaggle/input/datafortraining/Data.txt")
test_frames = propagate_tle(tle_test_data, timesteps=40)
test_frames = scaler.transform(test_frames.reshape(-1, test_frames.shape[-1])).reshape(test_frames.shape)
test_graphs = prepare_graph_data(test_frames)

predictions = generate_predictions(model, test_graphs)
denormalized_predictions = scaler.inverse_transform(predictions.reshape(-1, 3))

print("Running Complete")
