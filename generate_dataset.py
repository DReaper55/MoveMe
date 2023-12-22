import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define the number of orders and drivers
num_orders = 100
num_drivers = 20

# Generate synthetic data for orders
orders = pd.DataFrame({
    'OrderID': np.arange(1, num_orders + 1),
    'PickupLocation_X': np.random.rand(num_orders),
    'PickupLocation_Y': np.random.rand(num_orders),
    'DropoffLocation_X': np.random.rand(num_orders),
    'DropoffLocation_Y': np.random.rand(num_orders),
    'OrderType': np.random.choice(['product', 'food'], num_orders),
    'OrderPriority': np.random.choice(['express', 'standard'], num_orders),
    'Timestamp': [datetime.now() - timedelta(minutes=np.random.randint(1, 60)) for _ in range(num_orders)]
})

# Save the DataFrame to a JSON file
# orders.to_json('orders_data.json', orient='records', lines=True)

# Generate synthetic data for drivers
drivers = pd.DataFrame({
    'DriverID': np.arange(1, num_drivers + 1),
    'CurrentLocation_X': np.random.rand(num_drivers),
    'CurrentLocation_Y': np.random.rand(num_drivers),
    'Destination_X': np.random.rand(num_drivers),
    'Destination_Y': np.random.rand(num_drivers),
    'Availability': np.random.choice(['online', 'offline'], num_drivers),
    'OnlineStatus': np.random.choice(['standby', 'moving'], num_drivers),
    'Timestamp': [datetime.now() - timedelta(minutes=np.random.randint(1, 60)) for _ in range(num_drivers)],
    'CurrentOrders': np.random.uniform(0, 3.0, num_drivers),
    'TotalExpectedOrders': 3
})

# Save the DataFrame to a JSON file
drivers.to_json('drivers_data.json', orient='records', lines=True)

# Display sample orders and drivers dataframes
print("Sample Orders Data:")
print(orders.head())

print("\nSample Drivers Data:")
print(drivers.head())

# Merge orders and drivers dataframes based on timestamp
dataset = pd.merge_asof(
    orders.sort_values('Timestamp'),
    drivers.sort_values('Timestamp'),
    by='Timestamp',
    direction='nearest'
)

# Display the merged dataset
print("\nMerged Dataset:")
print(dataset)

# Save the DataFrame to a JSON file
# dataset.to_json('dataset_data.json', orient='records', lines=True)

# if __name__ == '__main__':
