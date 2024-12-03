import numpy as np
from sgp4.api import Satrec, jday
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def load_tle(file):
    tle_data = []
    satellite_names = []
    with open(file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines) - 2, 3):
            if i + 2 < len(lines):
                if not lines[i + 1].startswith("1") or not lines[i + 2].startswith("2"):
                    continue
                satellite_names.append(lines[i])
                tle_data.append((lines[i + 1], lines[i + 2]))
    print(f"Total TLE entries loaded: {len(tle_data)}")
    return tle_data, satellite_names

def propagate_tle(tle_data, timesteps=60):
    frames = []
    for tle in tle_data:
        sat = Satrec.twoline2rv(tle[0], tle[1])
        frame = []
        jd, fr = jday(2000, 1, 1, 0, 0, 0)

        for t in range(timesteps):
            jd_timestep = jd + t * (1)
            _, position, _ = sat.sgp4(jd_timestep, fr)
            if position:
                frame.append(position)
            else:
                frame.append([None, None, None])
        frames.append(frame)

    return np.array(frames)

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
        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs

class DGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def train_model(graphs, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGNN(input_dim=3, hidden_dim=16, output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for graph in graphs:
            graph = graph.to(device)
            optimizer.zero_grad()
            output = model(graph)
            loss = criterion(output, graph.x)
            loss.backward()
            optimizer.step()
    return model

def generate_predictions(model, graphs):
    model.eval()
    with torch.no_grad():
        predictions = []
        for i, graph in enumerate(graphs):
            output = model(graph).cpu().numpy()
            predictions.append(output)
    return np.array(predictions)


def create_graph_animation(frames, satellite_names, title, save_path, collision_threshold=1.0):
    collision_detected = False

    def create_graph(frames, timestep):
        nonlocal collision_detected
        G = nx.Graph()

        for i, pos in enumerate(frames[timestep]):
            if None not in pos:
                G.add_node(i, pos=pos, name=satellite_names[i])
                for j in range(i + 1, len(frames[timestep])):
                    if None not in frames[timestep][j]:
                        distance = np.linalg.norm(np.array(pos) - np.array(frames[timestep][j]))
                        if distance < collision_threshold:
                            print(f"Collision detected between {satellite_names[i]} and {satellite_names[j]} at timestep {timestep}")
                            collision_detected = True
                        G.add_edge(i, j)

        return G

    def animate(timestep):
        ax.clear()
        if timestep >= len(frames) - 1:
            if not collision_detected:
                print("No collision detected in the entire animation.")
            return
        G = create_graph(frames, timestep)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=100, node_color="blue", edge_color="black" ,font_color='white',font_size=7)

        ax.set_title(f"{title} | Timestep {timestep + 1}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ani = FuncAnimation(fig, animate, frames=min(frames.shape[1], len(frames)), interval=200)
    ani.save(save_path, writer="ffmpeg", fps=4)


tle_data, satellite_names = load_tle("/kaggle/input/datafortraining/Data.txt")
frames = propagate_tle(tle_data, timesteps=40)
graphs = prepare_graph_data(frames)

model = train_model(graphs, epochs=100)

tle_test_data, satellite_names = load_tle("/kaggle/input/short-dataset/Short Data.txt")
test_frames = propagate_tle(tle_test_data, timesteps=40)
test_graphs = prepare_graph_data(test_frames)
frame_list = generate_predictions(model, test_graphs)

create_graph_animation(frame_list, satellite_names, "DGNN Predictions", "dgnn_predictions.mp4")
