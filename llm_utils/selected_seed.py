import hashlib

def get_consistent_seed():
  """
  Selects a seed number from a list consistently across multiple runs.

  Returns:
    int: The selected seed number.
  """
  seed_list = [42, 1234, 9999, 2024, 2025]
  # Create a hash of the function name to ensure consistency
  hash_object = hashlib.sha256(b'get_consistent_seed')
  hash_value = int(hash_object.hexdigest(), 16)
  # Use the hash value to select a seed from the list
  seed_index = hash_value % len(seed_list)
  return seed_list[seed_index]

# Get the consistent seed
seed = get_consistent_seed()
print(f"Selected seed: {seed}")