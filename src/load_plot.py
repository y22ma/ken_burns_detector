#!/usr/bin/env python3
"""
Helper script to load and display interactive matplotlib figures saved as pickle files.
"""

import pickle
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python load_plot.py <path_to_pickle_file>")
        print("\nExample:")
        print("  python load_plot.py ken_burns_analysis.pkl")
        print("  python load_plot.py homography_parameters.pkl")
        sys.exit(1)
    
    pickle_file = sys.argv[1]
    
    try:
        with open(pickle_file, 'rb') as f:
            fig = pickle.load(f)
        
        print(f"Loaded figure from {pickle_file}")
        print("Displaying interactive plot. Close the window to exit.")
        
        # Show the figure
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: File '{pickle_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

