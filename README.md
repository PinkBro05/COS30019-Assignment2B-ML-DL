# COS30019 Assignment 2B - AI-Enhanced Traffic Navigation System

This project combines artificial intelligence search algorithms and machine learning for intelligent traffic navigation. It features an interactive GUI application that visualizes Melbourne traffic lights on a map and provides optimal pathfinding with real-time traffic predictions.

## Key Features

- **Interactive Map Visualization**: PyQt5-based GUI with Folium integration showing Melbourne traffic lights
- **Multiple Search Algorithms**: Comprehensive suite of AI search algorithms
- **Machine Learning Integration**: Deep learning models for traffic flow prediction
- **Real-time Path Optimization**: Combines search algorithms with ML predictions for time-optimal routes
- **Comparative Analysis**: Side-by-side algorithm performance evaluation

## Project Structure

```
├── main.py                     # Main GUI application with interactive map
├── Data/                       # Traffic data and graph files
│   ├── graph.txt              # Road network graph data
│   ├── Traffic_Lights.geojson # Melbourne traffic lights location data
│   └── temp_chunked_graph.txt # Temporary chunked graph for optimization
├── Search/                     # AI Search Algorithms Implementation
│   ├── search_utils.py        # Main search interface and utilities
│   ├── Uninformed_Search/     # BFS, DFS implementations
│   ├── Informed_Search/       # A*, Greedy Best-First Search
│   ├── Custom_Search/         # Dijkstra's Algorithm, Ant Colony Optimization
│   └── data_reader/           # Graph file parsing utilities
├── ML/                        # Machine Learning for Traffic Prediction
│   ├── predict_utils.py       # ML integration utilities
│   ├── Transformer/           # Transformer model for time series prediction
│   ├── LSTM/                  # LSTM model implementation
│   ├── GRU/                   # GRU model implementation
│   └── Utils/                 # Data preprocessing and feature engineering
└── Utils/                     # General utilities
    ├── create_graph.py        # Graph creation from traffic data
    ├── filter_chunk.py        # Graph chunking for performance optimization
    └── connect_components.py  # Graph connectivity analysis
```

## Getting Started

### Prerequisites

Install the required dependencies:

```powershell
pip install PyQt5 QWebEngineWidgets folium geopandas pandas numpy matplotlib scikit-learn torch
```

### Quick Start

1. **Launch the Interactive Application**:
   ```powershell
   python main.py
   ```

2. **Using the GUI**:
   - **Origin/Destination**: Enter SITE_NO values for traffic lights
   - **Algorithm Selection**: Choose from 6 different search algorithms
   - **Traffic Prediction**: Enable ML-based time predictions
   - **Results**: View multiple optimal paths with costs
   - **Visualization**: Click "Show on Map" to visualize paths

### Available Search Algorithms

| Algorithm | Code | Type | Use Case |
|-----------|------|------|----------|
| **A* Search** | AS | Informed | Optimal pathfinding with heuristics |
| **Greedy Best-First** | GBFS | Informed | Fast pathfinding (may not be optimal) |
| **Breadth-First Search** | BFS | Uninformed | Guaranteed shortest path (unweighted) |
| **Depth-First Search** | DFS | Uninformed | Memory efficient exploration |
| **Dijkstra's Algorithm** | CUS1 | Custom | Optimal shortest path (weighted graphs) |
| **Ant Colony Optimization** | CUS2 | Custom | Bio-inspired optimization |

## Machine Learning Integration

### Traffic Flow Prediction

The system includes three deep learning models for traffic flow prediction:

1. **Transformer Model** (Primary)
   - Attention-based architecture for sequence prediction
   - Best performance for traffic time series
   
2. **LSTM Model**
   - Long Short-Term Memory for temporal patterns
   
3. **GRU Model**
   - Gated Recurrent Units for efficient processing

### Using ML Predictions

1. **Enable Traffic Predictions**: Check the "Use Traffic Flow Predictions" checkbox
2. **Set Time**: Choose the start time for prediction
3. **Run Search**: The system will:
   - Use ML models to predict traffic flow
   - Convert predictions to time-based edge costs
   - Find time-optimal paths instead of distance-optimal

### Training ML Models

For detailed ML model training instructions, see [`ML/ML_README.md`](ML/ML_README.md).

**Quick training example**:
```powershell
cd ML
python Transformer/supervised_learning.py --data_file Data/Transformed/2024_final_time_series.csv
```

## Search Algorithm Usage

### Interactive GUI Usage

1. Launch the application: `python main.py`
2. Enter origin and destination SITE_NO values
3. Select algorithm from dropdown
4. Optionally enable traffic predictions
5. Click "Find Paths" to see results

### Programmatic Usage

```python
from Search.search_utils import find_paths

# Find paths using different algorithms
paths = find_paths('Data/graph.txt', origin='1001', destination='1002', algorithm='AS')

# With traffic predictions
from ML.predict_utils import prepare_traffic_based_search
if prepare_traffic_based_search('2024-01-01 09:00:00', '1001', '1002'):
    paths = find_paths('Data/temp_chunked_graph.txt', '1001', '1002', 'DIJK')
```

### Algorithm-Specific Features

**A* Search**:
- Optimal pathfinding with Manhattan distance heuristic
- Excellent performance on grid-like networks

**Ant Colony Optimization**:
- Bio-inspired optimization
- Configurable parameters (pheromone, evaporation rates)
- Supports parallel processing

**Dijkstra's Algorithm**:
- Guaranteed optimal shortest path
- Works with weighted graphs
- Good baseline algorithm

## GUI Features

### Interactive Map
- **Zoom/Pan**: Navigate Melbourne traffic network
- **Markers**: Click traffic lights for details
- **Path Visualization**: Routes highlighted in red
- **Origin/Destination**: Color-coded start/end points

### Search Panel
- **Real-time Input Validation**: Checks for valid SITE_NO values
- **Algorithm Comparison**: Switch between algorithms to compare results
- **Cost Display**: Shows both distance and time costs
- **Results Table**: Multiple paths ranked by optimality

### Performance Optimization
- **Graph Chunking**: Automatically reduces search space
- **Parallel Processing**: Multi-threaded ACO implementation
- **Caching**: Optimized data structures for repeated searches

## Data Sources

The project uses several Melbourne traffic datasets:

- **Traffic Signal Volume Data**: [VicRoads OpenData](https://opendata.transport.vic.gov.au/dataset/traffic-signal-volume-data)
- **Traffic Light Locations**: [VicGov Discover](https://discover.data.vic.gov.au/dataset/traffic-lights)
- **School Locations**: [VicGov Education Data](https://discover.data.vic.gov.au/dataset/school-locations-2024)
- **Public Holidays**: [Australian Government Data](https://data.gov.au/dataset/ds-dga-b1bc6077-dadd-4f61-9f8c-002ab2cdff10/details)

## Advanced Configuration

### Algorithm Parameters

**ACO Configuration**:
```python
# In Search/Custom_Search/aco_search.py
aco = ACO(
    graph=graph,
    ant_max_steps=500,
    num_iterations=100,
    evaporation_rate=0.1,
    alpha=0.7,  # Pheromone importance
    beta=0.3,   # Distance importance
)
```

**Graph Chunking**:
```python
# Adjust chunking aggressiveness
filtered_nodes, filtered_edges, _, _ = create_chunked_graph(
    graph_file_path, origin, destination, margin_factor=0.1  # Increase for larger search area
)
```

### Performance Tuning

- **Large Graphs**: Increase chunking margin_factor
- **Memory Constraints**: Use DFS instead of BFS
- **Speed Priority**: Use GBFS or reduce ACO iterations
- **Accuracy Priority**: Use Dijkstra or A* with tight heuristics

## Troubleshooting

### Common Issues

**"SITE_NO not found"**: 
- Check that traffic light data is properly loaded
- Verify SITE_NO format (usually 4-digit numbers)

**Search timeout**:
- Increase chunking margin_factor
- Use simpler algorithms (BFS/DFS) for initial testing

**ML prediction errors**:
- Ensure trained models exist in `ML/*/save_models/`
- Check data preprocessing completion

**GUI not loading**:
- Verify PyQt5 and QWebEngine installation
- Check file permissions for HTML generation

## Contributors

- **Hong Anh Nguyen** - Data Processing, Transformer Model, Search Algorithms
- **Phong Tran** - LSTM Model Implementation
- **James Luong** - GRU Model Implementation

## Academic Context

**Course**: COS30019 - Introduction to Artificial Intelligence  
**Assignment**: 2B - Machine Learning & Deep Learning Integration  
**Institution**: Swinburne University of Technology

## Requirements

- Python 3.8+
- PyQt5 with WebEngine
- PyTorch 1.8+
- Geopandas, Folium
- Pandas, NumPy, Matplotlib, Scikit-learn

## External Links

- [Project Dataset on HuggingFace](https://huggingface.co/datasets/PinkBro/vicroads-traffic-signals)
- [Detailed ML Documentation](ML/ML_README.md)

---