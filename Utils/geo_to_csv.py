import geopandas as gpd

# Load GeoJSON
gdf = gpd.read_file('Traffic_Lights.geojson')

# Optional: convert geometry to WKT text
gdf['geometry'] = gdf['geometry'].apply(lambda x: x.wkt)

# Save to CSV
gdf.to_csv('Traffic_Lights.csv', index=False)