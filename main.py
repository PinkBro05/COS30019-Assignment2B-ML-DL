"""
Script to plot traffic scats locations onto an interactive map.
Uses GeoPy for geographical calculations and Folium for visualization.
Includes search functionality to find paths between traffic light locations.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, Search
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import  QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSlider

# Import search utilities
from Search.search_utils import find_paths
# Import chunking utility
from Utils.filter_chunk import create_chunked_graph, write_chunked_graph_to_file

def read_geojson_file(geojson_file_path):
    """
    Read the GeoJSON file containing traffic light data with coordinates
    
    Args:
        geojson_file_path (str): Path to the GeoJSON file
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the traffic light data
    """
    try:
        # Read GeoJSON using geopandas
        gdf = gpd.read_file(geojson_file_path)
        print(f"Successfully read {len(gdf)} traffic light locations from {geojson_file_path}")
        
        # Extract longitude and latitude from geometry for easier access
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        
        # Display first few rows to verify
        # print(gdf[['SITE_NO', 'SITE_NAME', 'SITE_TYPE', 'longitude', 'latitude']].head())
        
        return gdf
    except Exception as e:
        print(f"Error reading GeoJSON file: {e}")
        sys.exit(1)

def filter_to_melbourne(gdf):
    """
    Filter the GeoDataFrame to include only traffic lights within Melbourne's bounding box
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with traffic light locations
        
    Returns:
        geopandas.GeoDataFrame: Filtered GeoDataFrame
    """
    # Melbourne bounding box (approximate)
    # Coordinates roughly cover Greater Melbourne area
    min_lat = -38.00
    max_lat = -37.50
    min_lon = 144.50
    max_lon = 145.50
    
    # Filter based on coordinates
    melbourne_gdf = gdf[(gdf['latitude'] >= min_lat) & 
                      (gdf['latitude'] <= max_lat) & 
                      (gdf['longitude'] >= min_lon) & 
                      (gdf['longitude'] <= max_lon)]
    
    print(f"Filtered from {len(gdf)} to {len(melbourne_gdf)} traffic lights within Melbourne area")
    
    return melbourne_gdf

def create_map(gdf, center=None):
    """
    Create an interactive map with markers for each traffic light location
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing locations with geometry
        center (tuple, optional): Center coordinates for the map (lat, lon). If None, uses the center of Melbourne.
        
    Returns:
        folium.Map: The created map object
    """
    if center is None:
        # Set center to CBD Melbourne
        center = [-37.8136, 144.9631]  # Melbourne CBD coordinates
      # Create a map centered on Melbourne with appropriate zoom
    m = folium.Map(
        location=center,
        zoom_start=12,  # Increased zoom level for Melbourne
        max_zoom=22,  # Increase maximum zoom level
        tiles='CartoDB positron',  # Use a minimal, faster loading tile layer
        control_scale=True  # Add scale for better navigation
    )
    
    # Create a marker cluster with optimized settings for performance
    marker_cluster = MarkerCluster(
        options={
            'disableClusteringAtZoom': 16,  # Stop clustering at high zoom levels
            'maxClusterRadius': 100,  # Increase clustering radius for even fewer markers at low zoom
            'spiderfyOnMaxZoom': True,  # Spread out markers when clicking on a cluster at max zoom
            'chunkedLoading': True  # Enable chunked loading for better performance
        }
    ).add_to(m)
    
    # Add markers for each location directly to the cluster without GeoJSON layer
    # This improves performance significantly
    for idx, row in gdf.iterrows():
        # Create popup text with relevant information
        popup_text = f"Site No: {row['SITE_NO']}<br>Name: {row['SITE_NAME']}<br>Type: {row['SITE_TYPE']}"
        if pd.notna(row['COMMENTS']) and row['COMMENTS']:
            popup_text += f"<br>Comments: {row['COMMENTS']}"
            
        # Add circle marker to cluster - more efficient than regular markers
        circle_marker = folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            popup=popup_text,
            radius=4,
            color='#3186cc',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            tooltip=f"Site: {row['SITE_NO']} - {row['SITE_NAME']}"  # Add tooltip for hover information
        )
        circle_marker.add_to(marker_cluster)
    
    # Create a search control for the map
    create_search(gdf, m)

    # Add a layer control to toggle the visibility
    folium.LayerControl(
        position='topright',
        collapsed=True,
    ).add_to(m)
        
    return m

def create_search(gdf, map_obj):
    """
    Create a search control for the map to find traffic light locations by name
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing locations with geometry
        map_obj (folium.Map): The map object to add the search control to
        
    Returns:
        None
    """
    # Create a search control for the map but ensure it's not visible initially
    scats_layer = folium.FeatureGroup(name='Traffic Lights', show=False)
    scats = folium.GeoJson(
        gdf,
        name='Traffic Lights',
        show=False,
    )
    
    # Add search functionality
    search = Search(
        layer=scats,
        geom_type='Point',
        placeholder='Search by Site No or Name',
        collapsed=True,
        search_label='SITE_NO',
        position='topright'
    ).add_to(map_obj)

    # Add the layer to the map
    scats.add_to(scats_layer)
    scats_layer.add_to(map_obj)

class MainWindow(QMainWindow):
    """Main application window with map view and search panel"""
    
    def __init__(self, gdf):
        super().__init__()
        
        # Store the traffic light data
        self.gdf = gdf
        
        # Initialize UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Melbourne Traffic Light Locations with Path Search")
        self.setGeometry(100, 100, 1280, 768)  # Larger window to accommodate search panel
        
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Create and configure the map widget
        self.map_widget = QtWebEngineWidgets.QWebEngineView()
        
        # Create search panel
        search_panel = self.create_search_panel()
        
        # Add widgets to main layout
        main_layout.addWidget(search_panel, 1)
        main_layout.addWidget(self.map_widget, 3)
        
        # Generate and load the map
        self.map_obj = create_map(self.gdf)
        output_file = 'map_output.html'
        self.map_obj.save(output_file)
        self.map_widget.load(QtCore.QUrl.fromLocalFile(os.path.abspath(output_file)))
        
        self.setCentralWidget(main_widget)
        
    def create_search_panel(self):
        """Create the search panel for path finding"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Create input group box
        input_group = QGroupBox("Search for Paths")
        input_layout = QVBoxLayout()
        
        # Origin input
        origin_layout = QHBoxLayout()
        origin_label = QLabel("Origin (SITE_NO):")
        self.origin_input = QLineEdit()
        origin_layout.addWidget(origin_label)
        origin_layout.addWidget(self.origin_input)
        
        # Destination input
        dest_layout = QHBoxLayout()
        dest_label = QLabel("Destination (SITE_NO):")
        self.dest_input = QLineEdit()
        dest_layout.addWidget(dest_label)
        dest_layout.addWidget(self.dest_input)
        
        # Algorithm selection
        algo_layout = QHBoxLayout()
        algo_label = QLabel("Algorithm:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "A* Search (AS)", 
            "Greedy Best-First Search (GBFS)", 
            "Breadth-First Search (BFS)", 
            "Depth-First Search (DFS)", 
            "Dijkstra's Algorithm (CUS1)", 
            "Ant Colony Optimization (CUS2)"
        ])
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.algo_combo)
        
        # Search button
        search_button = QPushButton("Find Paths")
        search_button.clicked.connect(self.find_paths)
        
        # Add components to input layout
        input_layout.addLayout(origin_layout)
        input_layout.addLayout(dest_layout)
        input_layout.addLayout(algo_layout)
        input_layout.addWidget(search_button)
        
        # Set input group layout
        input_group.setLayout(input_layout)
        
        # Create results table
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Path", "Cost (meters)", "Show"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        
        # Add components to main panel layout
        layout.addWidget(input_group)
        layout.addWidget(results_group)
        layout.addStretch(1)
        
        panel.setLayout(layout)
        return panel
    
    def find_paths(self):
        """Find paths between origin and destination"""
        # Get inputs
        origin = self.origin_input.text().strip()
        destination = self.dest_input.text().strip()
        
        # Basic validation
        if not origin or not destination:
            QtWidgets.QMessageBox.warning(
                self, "Input Error", 
                "Please enter both Origin and Destination SITE_NO values."
            )
            return
            
        # Validate origin and destination are valid SITE_NO values
        site_nos = set(self.gdf['SITE_NO'].astype(str))
        if origin not in site_nos:
            QtWidgets.QMessageBox.warning(
                self, "Input Error", 
                f"Origin '{origin}' is not a valid SITE_NO."
            )
            return
            
        if destination not in site_nos:
            QtWidgets.QMessageBox.warning(
                self, "Input Error", 
                f"Destination '{destination}' is not a valid SITE_NO."
            )
            return
              # Get selected algorithm code
        algo_text = self.algo_combo.currentText()
        if "AS" in algo_text:
            algorithm = "AS"
        elif "GBFS" in algo_text:
            algorithm = "GBFS"
        elif "BFS" in algo_text:
            algorithm = "BFS"
        elif "DFS" in algo_text:
            algorithm = "DFS"
        elif "CUS1" in algo_text:
            algorithm = "DIJK"  # Map to Dijkstra's algorithm code in search_utils.py
        else:
            algorithm = "ACO"   # Map to ACO algorithm code in search_utils.py
            
        # Show loading message
        self.statusBar().showMessage(f"Finding paths with {algorithm}...")
        QtWidgets.QApplication.processEvents()        
        
        try:
            # Original graph file path
            graph_file_path = os.path.join('Data', 'graph.txt')
            
            # Use chunking to reduce search space
            self.statusBar().showMessage(f"Applying chunking to reduce search space...")
            QtWidgets.QApplication.processEvents()
            
            try:
                # Apply chunking to create filtered graph
                filtered_nodes, filtered_edges, chunked_origin, chunked_destinations = create_chunked_graph(
                    graph_file_path, origin, destination, margin_factor=0.5
                )
                
                # Create temporary chunked graph file
                temp_graph_path = os.path.join('Data', 'temp_chunked_graph.txt')
                write_chunked_graph_to_file(
                    filtered_nodes, filtered_edges, chunked_origin, chunked_destinations, temp_graph_path
                )
                
                # Find paths using the chunked graph
                self.statusBar().showMessage(f"Finding paths with {algorithm} on reduced graph...")
                QtWidgets.QApplication.processEvents()
                paths = find_paths(temp_graph_path, origin, destination, algorithm, top_k=5)
            except Exception as chunk_error:
                # Fallback to original graph if chunking fails
                self.statusBar().showMessage(f"Chunking failed, using full graph: {str(chunk_error)}")
                QtWidgets.QApplication.processEvents()
                paths = find_paths(graph_file_path, origin, destination, algorithm, top_k=5)
            
            # Display results in the table
            self.display_results(paths)
            
            # Update status
            self.statusBar().showMessage(f"Found {len(paths)} paths using {algorithm}.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Search Error", 
                f"An error occurred while searching for paths: {str(e)}"
            )
            self.statusBar().showMessage("Search failed.")
    
    def display_results(self, paths):
        """Display search results in the table"""
        # Clear previous results
        self.results_table.setRowCount(0)
        
        # Add new results
        for i, (path, cost) in enumerate(paths):
            self.results_table.insertRow(i)
            
            # Path
            path_str = " → ".join(path)
            self.results_table.setItem(i, 0, QTableWidgetItem(path_str))
            
            # Cost
            cost_item = QTableWidgetItem(str(int(cost)))
            cost_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.results_table.setItem(i, 1, cost_item)
            # Show button
            show_button = QPushButton("Show on Map")
            show_button.clicked.connect(lambda checked, p=path, c=cost: self.show_path_on_map(p))
            self.results_table.setCellWidget(i, 2, show_button)
    
    def show_path_on_map(self, path):
        # TODO Implement the logic to highlight the path on the map and zoom in on it, fixing bug for AS and GBFS
        pass

def main():
    """
    Main function to read data and create the map visualization with search functionality
    """
    # Path to the GeoJSON file
    geojson_path = os.path.join('Data', 'Traffic_Lights.geojson')
    
    # Read the GeoJSON file
    gdf = read_geojson_file(geojson_path)
    
    # Filter to Melbourne area
    gdf = filter_to_melbourne(gdf)
    
    # Create the PyQt5 application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create the main window
    main_window = MainWindow(gdf)
    main_window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()