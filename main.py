"""
Script to plot traffic scats locations onto an interactive map.
Uses GeoPy for geographical calculations and Folium for visualization.
Includes search functionality to find paths between traffic light locations.
"""

import sys
import os
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, Search
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
from PyQt5.QtWidgets import  QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QSlider, QDateTimeEdit
from PyQt5.QtCore import QDateTime
import traceback

# Import search utilities
from Search.search_utils import find_paths
# Import chunking utility
from Utils.filter_chunk import create_chunked_graph, write_chunked_graph_to_file
# Import traffic prediction utilities
from ML.predict_utils import prepare_traffic_based_search

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

    # Add geocoder
    folium.plugins.Geocoder().add_to(m)

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
        
        # Start time input for traffic prediction
        time_layout = QHBoxLayout()
        time_label = QLabel("Start Time:")
        self.time_input = QDateTimeEdit()
        self.time_input.setDateTime(QDateTime.currentDateTime())
        self.time_input.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.time_input.setCalendarPopup(True)
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.time_input)
        
        # Use traffic prediction checkbox
        traffic_layout = QHBoxLayout()
        self.use_traffic_checkbox = QCheckBox("Use Traffic Flow Predictions")
        self.use_traffic_checkbox.setChecked(False)
        self.use_traffic_checkbox.setToolTip("Enable to use ML-predicted traffic flow for time-based costs")
        traffic_layout.addWidget(self.use_traffic_checkbox)
        
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
        input_layout.addLayout(time_layout)
        input_layout.addLayout(traffic_layout)
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
        start_time = self.time_input.dateTime().toString("yyyy-MM-dd hh:mm:ss")
        use_traffic = self.use_traffic_checkbox.isChecked()
        
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
        
        # If traffic prediction is enabled, get ML predictions
        traffic_data = None
        if use_traffic:
            try:
                print(f"Getting traffic predictions for time: {start_time}")
                traffic_data = get_traffic_predictions(start_time, origin, destination)
                if traffic_data['status'] == 'success':
                    print(f"Successfully obtained traffic predictions for {len(traffic_data['time_costs'])} edges")
                else:
                    print(f"Traffic prediction failed: {traffic_data.get('error', 'Unknown error')}")
                    QtWidgets.QMessageBox.warning(
                        self, "Traffic Prediction Warning",
                        f"Traffic prediction failed. Using distance-based costs.\nError: {traffic_data.get('error', 'Unknown error')}"
                    )
                    traffic_data = None
            except Exception as e:
                print(f"Error getting traffic predictions: {e}")
                QtWidgets.QMessageBox.warning(
                    self, "Traffic Prediction Error",
                    f"Failed to get traffic predictions. Using distance-based costs.\nError: {str(e)}"
                )
                traffic_data = None
        
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
                    graph_file_path, origin, destination, margin_factor=0.1
                )
                
                # If traffic prediction is enabled, modify edge costs
                if traffic_data and traffic_data['status'] == 'success':
                    self.statusBar().showMessage(f"Applying traffic-based costs...")
                    QtWidgets.QApplication.processEvents()
                    filtered_edges = self._apply_traffic_costs(filtered_edges, traffic_data['time_costs'])
                
                # Create temporary chunked graph file
                temp_graph_path = os.path.join('Data', 'temp_chunked_graph.txt')
                write_chunked_graph_to_file(
                    filtered_nodes, filtered_edges, chunked_origin, chunked_destinations, temp_graph_path
                )
                
                # Find paths using the chunked graph
                cost_type = "time (seconds)" if traffic_data and traffic_data['status'] == 'success' else "distance (meters)"
                self.statusBar().showMessage(f"Finding paths with {algorithm} using {cost_type}...")
                QtWidgets.QApplication.processEvents()
                paths = find_paths(temp_graph_path, origin, destination, algorithm, top_k=5)
                
                # Store cost type for display
                self.current_cost_type = cost_type
                
            except Exception as chunk_error:
                # Fallback to original graph if chunking fails
                self.statusBar().showMessage(f"Chunking failed, using full graph: {str(chunk_error)}")
                QtWidgets.QApplication.processEvents()
                paths = find_paths(graph_file_path, origin, destination, algorithm, top_k=5)
                self.current_cost_type = "distance (meters)"
            
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
        
        # Determine if we need to show both time and distance
        is_time_based = hasattr(self, 'current_cost_type') and 'time' in self.current_cost_type
        
        # Update columns based on cost type
        if is_time_based:
            # Show both time and distance columns when using traffic prediction
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels(["Path", "Time", "Distance (m)", "Show"])
            self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        else:
            # Use original 3-column layout for distance-only results
            self.results_table.setColumnCount(3)
            self.results_table.setHorizontalHeaderLabels(["Path", "Distance (m)", "Show"])
            self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Add new results
        for i, (path, cost) in enumerate(paths):
            self.results_table.insertRow(i)
            
            # Path
            path_str = " → ".join(path)
            self.results_table.setItem(i, 0, QTableWidgetItem(path_str))
            
            # Calculate path distance if we're in time-based mode
            # This is done by estimating from the path nodes using straight-line distances
            path_distance = 0
            if is_time_based and len(path) > 1:
                for j in range(len(path) - 1):
                    # Get coordinates for both points
                    row1 = self.gdf[self.gdf['SITE_NO'].astype(str) == str(path[j])].iloc[0]
                    row2 = self.gdf[self.gdf['SITE_NO'].astype(str) == str(path[j+1])].iloc[0]
                    
                    # Calculate Euclidean distance and convert to approximate meters
                    lat1, lon1 = row1['latitude'], row1['longitude']
                    lat2, lon2 = row2['latitude'], row2['longitude']
                    
                    # Simple haversine distance calculation (approximate)
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371000  # Earth radius in meters
                    
                    dLat = radians(lat2 - lat1)
                    dLon = radians(lon2 - lon1)
                    a = sin(dLat/2) * sin(dLat/2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2) * sin(dLon/2)
                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                    distance = R * c
                    
                    path_distance += distance
            
            if is_time_based:
                # Format time in minutes and seconds
                minutes = int(cost) // 60
                seconds = int(cost) % 60
                time_display = f"{minutes} min {seconds} sec"
                
                # Add time display
                time_item = QTableWidgetItem(time_display)
                time_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.results_table.setItem(i, 1, time_item)
                
                # Add distance display
                distance_item = QTableWidgetItem(f"{int(path_distance)}")
                distance_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.results_table.setItem(i, 2, distance_item)
                
                # Show button in column 3
                show_button = QPushButton("Show on Map")
                show_button.clicked.connect(lambda checked, p=path, c=cost: self.show_path_on_map(p))
                self.results_table.setCellWidget(i, 3, show_button)
            else:
                # For distance-only mode, just show the distance in meters (column 1)
                distance_item = QTableWidgetItem(f"{int(cost)}")
                distance_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.results_table.setItem(i, 1, distance_item)
                
                # Show button in column 2
                show_button = QPushButton("Show on Map")
                show_button.clicked.connect(lambda checked, p=path, c=cost: self.show_path_on_map(p))
                self.results_table.setCellWidget(i, 2, show_button)
    
    def show_path_on_map(self, path):
        # Retrieve coordinates for each site in the path
        coords = []
        for site in path:
            # Match SITE_NO as string to handle numeric/string types
            row = self.gdf[self.gdf['SITE_NO'].astype(str) == str(site)].iloc[0]
            coords.append([row['latitude'], row['longitude']])
            
        # Recreate base map with all markers
        self.map_obj = create_map(self.gdf)
        
        # Add markers for Origin and Destination
        if coords:
            # Add origin marker
            folium.Marker(
                location=coords[0],
                popup=f"Origin: {path[0]}",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(self.map_obj)
            
            # Add destination marker
            folium.Marker(
                location=coords[-1],
                popup=f"Destination: {path[-1]}",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(self.map_obj)
                
        # Add the path as a red polyline
        folium.PolyLine(coords, color='red', weight=5, opacity=0.8).add_to(self.map_obj)
        
        # Adjust map to fit the path bounds
        self.map_obj.fit_bounds(coords)
        
        # Save and reload the updated map
        output_file = 'map_output.html'
        self.map_obj.save(output_file)
        self.map_widget.load(QtCore.QUrl.fromLocalFile(os.path.abspath(output_file)))

    def _apply_traffic_costs(self, edges, traffic_costs):
        """
        Apply traffic-based time costs to graph edges.
        
        Args:
            edges: Dictionary of graph edges
            traffic_costs: Dictionary of time-based costs from ML prediction
            
        Returns:
            Modified edges dictionary with updated costs
        """
        modified_edges = edges.copy()
        applied_count = 0
        
        for edge_key, edge_data in modified_edges.items():
            # Check if we have a time cost for this edge
            if edge_key in traffic_costs:
                # Update the cost/weight with time-based cost
                original_cost = edge_data
                time_cost = traffic_costs[edge_key]
                
                # Replace the edge cost with time cost
                modified_edges[edge_key] = time_cost
                applied_count += 1
                
                print(f"Updated edge {edge_key}: {original_cost} -> {time_cost}")
        
        print(f"Applied time-based costs to {applied_count} out of {len(edges)} edges")
        return modified_edges

def get_traffic_predictions(start_time, origin, destination):
    """
    Get traffic predictions for a specified start time, origin and destination.
    
    Args:
        start_time (str): Start time for prediction in format 'YYYY-MM-DD HH:MM:SS'
        origin (str): Origin node ID
        destination (str): Destination node ID
        
    Returns:
        dict: Dictionary containing prediction results with keys:
            - status: 'success' or 'error'
            - time_costs: Dictionary mapping edge tuples to travel time costs (if successful)
            - error: Error message (if failed)
    """
    try:
        # Create temporary chunked graph file path
        chunked_graph_path = os.path.join('Data', 'temp_chunked_graph.txt')
        output_path = os.path.join('Data', 'traffic_graph.txt')
        
        # Prepare graph with traffic predictions
        updated_graph_path = prepare_traffic_based_search(
            origin, destination, start_time, 
            chunked_graph_path=chunked_graph_path, 
            output_path=output_path
        )
        
        if updated_graph_path:
            # Parse the updated graph to extract edge costs
            # Import parser for reading the graph file
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "parser", 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Search', 'data_reader', 'parser.py')
            )
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            parse_graph_file = parser_module.parse_graph_file
            
            # Parse the updated graph file to get edges with new costs
            _, edges, _, _ = parse_graph_file(updated_graph_path)
            
            # Format edges as tuples for compatibility with the apply_traffic_costs method
            time_costs = {}
            for edge_str, cost in edges.items():
                # Convert edge string to tuple if it's not already
                if isinstance(edge_str, tuple):
                    node1, node2 = edge_str
                else:
                    # This shouldn't happen with the current implementation, but added for robustness
                    parts = edge_str.strip('()').split(',')
                    node1, node2 = parts[0], parts[1]
                
                time_costs[(node1, node2)] = cost
            
            return {
                'status': 'success',
                'time_costs': time_costs
            }
        else:            
            return {
                'status': 'error',
                'error': 'Failed to prepare traffic-based search graph'
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Error in traffic prediction: {str(e)}\n{traceback.format_exc()}"
        }

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