"""
create venv: 
- uv venv
- source .venv/bin/activate  
- uv pip install {list of libs}
"""

from typing import Dict
import numpy as np
import ray
import unittest

# Create datasets from on-disk files, Python objects, and cloud storage like S3.
ds = ray.data.read_csv("s3://anonymous@ray-example-data/iris.csv")

# Apply functions to transform data. Ray Data executes transformations in parallel.
def compute_area(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    length = batch["petal length (cm)"]
    width = batch["petal width (cm)"]
    batch["petal area (cm^2)"] = length * width
    return batch

transformed_ds = ds.map_batches(compute_area)

# Iterate over batches of data.
for batch in transformed_ds.iter_batches(batch_size=4):
    print(batch)

# Save dataset contents to on-disk files or cloud storage.
transformed_ds.write_parquet("local:///tmp/iris/")

# Read the Parquet file back into a Ray dataset.
loaded_ds = ray.data.read_parquet("local:///tmp/iris/")

# Iterate over batches of data and print them.
for batch in loaded_ds.iter_batches(batch_size=4):
    print(batch)

class TestComputeArea(unittest.TestCase):
    def test_compute_area(self):
        # Sample input
        batch = {
            "petal length (cm)": np.array([1.0, 2.0, 3.0]),
            "petal width (cm)": np.array([0.5, 1.0, 1.5])
        }
        expected_output = {
            "petal length (cm)": np.array([1.0, 2.0, 3.0]),
            "petal width (cm)": np.array([0.5, 1.0, 1.5]),
            "petal area (cm^2)": np.array([0.5, 2.0, 4.5])
        }
        
        # Call the function
        result = compute_area(batch)
        
        # Assert the result
        np.testing.assert_array_equal(result["petal area (cm^2)"], expected_output["petal area (cm^2)"])

if __name__ == '__main__':
    unittest.main()
