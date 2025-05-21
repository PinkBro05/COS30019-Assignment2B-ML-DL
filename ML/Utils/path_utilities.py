import os

class PathManager:
    """Handles path construction and validation for the preprocessing pipeline."""
    
    @staticmethod
    def get_project_root():
        """Get the project root directory."""
        # Start from the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If we're in Utils, go one level up
        if os.path.basename(current_dir) == "Utils":
            return os.path.dirname(current_dir)
        return current_dir
    
    @staticmethod
    def normalize_path(path):
        """Convert path to standard format."""
        return os.path.normpath(path)
    
    @staticmethod
    def resolve_data_path(base_dir=None, data_subpath=None):
        """Resolve the path to the data directory.
        
        Args:
            base_dir: Base directory (defaults to project root)
            data_subpath: Subpath within Data directory
            
        Returns:
            Normalized path to the data directory
        """
        if base_dir is None:
            base_dir = PathManager.get_project_root()
        
        # Check if base_dir already ends with 'Data'
        if os.path.basename(base_dir) == "Data":
            if data_subpath:
                return PathManager.normalize_path(os.path.join(base_dir, data_subpath))
            return base_dir
        
        # Otherwise, append Data to the path
        data_dir = os.path.join(base_dir, "Data")
        if data_subpath:
            return PathManager.normalize_path(os.path.join(data_dir, data_subpath))
        return data_dir
    
    @staticmethod
    def ensure_dir_exists(dir_path):
        """Ensure directory exists, create if necessary."""
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    @staticmethod
    def get_raw_dir(base_dir=None):
        """Get path to the Raw/main directory."""
        return PathManager.resolve_data_path(base_dir, "Raw/main")
    
    @staticmethod
    def get_transformed_dir(base_dir=None):
        """Get path to the Transformed directory."""
        return PathManager.resolve_data_path(base_dir, "Transformed")
    
    @staticmethod
    def find_file(filename, search_dirs):
        """Find a file in a list of directories.
        
        Args:
            filename: Name of the file to find
            search_dirs: List of directories to search
            
        Returns:
            Full path to the file if found, None otherwise
        """
        for dir_path in search_dirs:
            file_path = os.path.join(dir_path, filename)
            if os.path.exists(file_path):
                return file_path
        return None